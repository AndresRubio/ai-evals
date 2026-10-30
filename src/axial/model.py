"""The one place that knows how to call a model.

Everything else takes a `Model`. That is what lets every test in this project run
offline against `FakeModel`, and what keeps retry, cost accounting and the
served-model check in a single place instead of scattered through the pipeline.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeGuard

from pydantic import BaseModel

from axial.errors import AxialError


class ServedModelMismatch(AxialError):
    """The provider served a different model than the one requested.

    A score produced by the wrong model measures nothing, and a silent provider
    fallback may not surface anywhere else.
    """


@dataclass(frozen=True)
class Completion:
    """A model response.

    `stop_reason` values follow litellm's normalised OpenAI shape (e.g. "stop",
    "length"), not Anthropic-native naming, since `LiteLLMModel` is the only
    real producer of this field and never emits the latter.
    """

    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "stop"
    truncated: bool = False
    attempts: int = 1
    latency_s: float = 0.0
    # Only a decision model fills this in. A text model has no calibrated
    # probability to report, and `None` says so rather than implying 0.0.
    probability: float | None = None


_DATED_SNAPSHOT_SUFFIX = re.compile(r"^(?P<base>.+)-\d{8}$")


def _normalise_model_id(model_id: str) -> str:
    """Strip a provider/routing prefix from a model id.

    `anthropic/claude-opus-5`, `bedrock.claude-opus-5` and
    `us.anthropic.claude-opus-5` all reduce to `claude-opus-5`.

    A dotted prefix is only stripped when it is purely alphabetic, because a dot
    also separates a version: splitting `jev-1.13` on its last `.` leaves `13`, and
    then `jev-1.13` and `jev-2.13` compare equal — defeating, for every
    dotted-version model, the substitution check this function exists to serve.
    """
    tail = model_id.rsplit("/", 1)[-1]
    while True:
        prefix, dot, rest = tail.partition(".")
        if not (dot and rest and prefix.isalpha()):
            return tail
        tail = rest


def _served_model_matches(requested: str, served: str) -> bool:
    """Check whether a served model id is an acceptable match for the requested one.

    A bare prefix match (`served.startswith(requested)`) is not enough: it lets a
    genuinely different model through whenever it shares a prefix, e.g. requested
    `gpt-4` served `gpt-4o`, or requested `claude-opus-5` served
    `claude-opus-5-legacy` — exactly the substitution this check exists to catch.
    So both sides are normalised the same way and compared for exact equality,
    with one documented exception: a served id equal to the requested id plus a
    trailing `-YYYYMMDD` is an accepted alias-to-snapshot resolution.
    """
    norm_requested = _normalise_model_id(requested)
    norm_served = _normalise_model_id(served)
    if norm_served == norm_requested:
        return True
    match = _DATED_SNAPSHOT_SUFFIX.match(norm_served)
    return match is not None and match.group("base") == norm_requested


class Model(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        schema: type[BaseModel] | None = None,
    ) -> Completion: ...


class FakeModel:
    """Deterministic model for tests and for the offline worked example."""

    def __init__(
        self,
        responses: dict[str, str] | Callable[[str], str],
        *,
        model: str = "fake",
        fail_times: int = 0,
        truncated: bool = False,
    ) -> None:
        self._responses = responses
        self._model = model
        self._fail_times = fail_times
        self._truncated = truncated
        self.calls: list[str] = []
        self.schemas: list[type[BaseModel] | None] = []

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        self.calls.append(prompt)
        self.schemas.append(schema)
        attempts = self._fail_times + 1
        if callable(self._responses):
            text = self._responses(prompt)
        else:
            text = self._responses[prompt]
        return Completion(
            text=text,
            model=self._model,
            usage={"input_tokens": max(1, len(prompt) // 4), "output_tokens": max(1, len(text) // 4)},
            stop_reason="length" if self._truncated else "stop",
            truncated=self._truncated,
            attempts=attempts,
        )


class LiteLLMModel:
    """Real model access. The only importer of litellm in the codebase."""

    def __init__(
        self,
        model: str,
        *,
        max_retries: int = 3,
        timeout_s: float = 120.0,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        # `seed=None` (the production default) seeds `random.Random` from OS entropy,
        # so concurrent instances draw independent backoff sequences instead of
        # retrying in lockstep against a shared rate limit. Pass an int only in
        # tests, where a reproducible sequence is the point.
        self._rng = random.Random(seed)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        import litellm

        # Ask for a schema-validated response where the provider can do it. Where it
        # cannot, the caller's text parse still runs — which is why that parse stays.
        extra: dict[str, object] = {}
        if schema is not None:
            try:
                supported = litellm.supports_response_schema(model=self.model)
            except Exception:  # noqa: BLE001 - an old litellm, or an unknown model
                supported = False
            if supported:
                extra["response_format"] = schema

        last_error: Exception | None = None
        started = time.monotonic()
        for attempt in range(1, self.max_retries + 1):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    timeout=self.timeout_s,
                    **extra,
                )
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if terminal
                last_error = exc
                if attempt == self.max_retries:
                    break
                # Jittered backoff: an unjittered retry loop turns one rate limit
                # into a synchronised stampede.
                time.sleep((2 ** (attempt - 1)) * (0.5 + self._rng.random()))
                continue

            served = getattr(response, "model", self.model) or self.model
            if not _served_model_matches(self.model, served):
                raise ServedModelMismatch(
                    f"requested {self.model!r} but the provider served {served!r}"
                )

            choice = response.choices[0]
            stop_reason = getattr(choice, "finish_reason", "stop") or "stop"
            usage = getattr(response, "usage", None)
            # Invariant 9: no silent zeros. A response with no `usage` at all, or a
            # `usage` object that carries neither token count, means the cost is
            # unknown -- not zero -- so record the absence as an empty dict rather
            # than fabricating `{"input_tokens": 0, "output_tokens": 0}`. `cost_usd`
            # (checks.py) treats an empty usage dict the same way it treats an
            # unpriced model: `None`, not a silently-wrong $0.00.
            input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            output_tokens = (
                getattr(usage, "completion_tokens", None) if usage is not None else None
            )
            usage_dict: dict[str, int] = (
                {}
                if input_tokens is None and output_tokens is None
                else {"input_tokens": input_tokens or 0, "output_tokens": output_tokens or 0}
            )
            return Completion(
                text=choice.message.content or "",
                model=served,
                usage=usage_dict,
                stop_reason=stop_reason,
                truncated=stop_reason == "length",
                attempts=attempt,
                latency_s=time.monotonic() - started,
            )

        raise AxialError(f"model call failed after {self.max_retries} attempts: {last_error}")


DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# Providers whose models answer typed questions instead of writing text. They reach
# a different endpoint entirely, so this is not a preference — a text call to one of
# these fails, and a decision call to a text model is not a thing that exists.
_DECISION_PROVIDERS = frozenset({"typesafe"})


def is_probability(value: object) -> TypeGuard[float]:
    """Whether `value` is a number from 0 to 1.

    A bool is an int to Python and a numeric string would pass a cast, so both are
    refused by type; `0 <= v <= 1` is written this way round so that nan fails it.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= value <= 1.0
    )


def is_decision_model(model_id: str) -> bool:
    """Whether `model_id` names a decision model rather than a text model."""
    segments = [seg for seg in model_id.lstrip("~").split("/") if seg]
    return any(seg in _DECISION_PROVIDERS for seg in segments[:-1])


def require_text_model(model_id: str) -> str:
    """Refuse a decision model where generated prose is the point.

    Open coding and taxonomy proposal need sentences. A decision model returns a
    typed answer and nothing else, so pointing one at those commands would produce
    an empty taxonomy rather than an error — which is the failure mode this exists
    to convert into a loud one.
    """
    if is_decision_model(model_id):
        raise AxialError(
            f"{model_id!r} is a decision model: it cannot generate text. This step "
            "needs prose, so set `judge_model` to a text model for it."
        )
    return model_id

_FLOATING_ALIAS = re.compile(r"(^~)|(\blatest\b)")


def _require_pinned(model_id: str) -> str:
    """Refuse a model id that can move underneath a promoted judge.

    `promote` binds a PROMOTED marker to `judge_yaml_sha256` and
    `verdicts_sha256`. Both hashes cover the judge definition and the verdicts it
    earned — neither covers which weights the provider served. A floating alias
    such as `~typesafe/jev-latest` therefore swaps the judge while every recorded
    hash still matches, and the gate keeps attesting to an alignment run that was
    measured on a different model. Pin the version and bump it deliberately.
    """
    if _FLOATING_ALIAS.search(model_id):
        raise AxialError(
            f"{model_id!r} is a floating alias; a judge model must be pinned to an "
            "exact version (e.g. 'typesafe/jev-1.13'). An alias that moves changes "
            "the judge without changing judge.yaml, so a PROMOTED marker would keep "
            "verifying against hashes it no longer describes."
        )
    return model_id


class JevModel:
    """TypeSafe's Jev behind the `Model` protocol, via OpenRouter's Decisions API.

    Jev is a decision model, not a text model: it answers typed questions about a
    state and returns calibrated probabilities. It does not run on the
    OpenAI-compatible chat endpoint, so litellm cannot reach it and `LiteLLMModel`
    is not an option — this class speaks the Decisions API directly.

    The adapter asks exactly one `noul` (yes/no probability) question per call and
    thresholds the answer into the same `{"reason": ..., "verdict": ...}` JSON an
    LLM judge is asked to emit. That keeps `run_judge` and `parse_verdict`
    unchanged, and it keeps the prompt that `assemble_prompt` built — few-shot
    train-only guarantee included — as the state Jev reads.

    What you give up is the critique. Jev returns a number and nothing to read, so
    the probability and the threshold are recorded as the reason: it is the whole
    audit trail. Use this for a high-volume gate, not for the alignment pass where
    you need to see *why* the judge and a human disagree.
    """

    def __init__(
        self,
        model: str = "typesafe/jev-1.13",
        *,
        instructions: str,
        true_when: str,
        false_when: str,
        threshold: float = 0.5,
        max_retries: int = 3,
        timeout_s: float = 60.0,
        seed: int | None = None,
    ) -> None:
        self.model = _require_pinned(model)
        self.instructions = instructions
        self.true_when = true_when
        self.false_when = false_when
        self.threshold = threshold
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._rng = random.Random(seed)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        """Grade `prompt` and return the verdict as the JSON the judge already parses.

        `max_tokens` and `schema` are accepted to satisfy the `Model` protocol and
        are deliberately unused: Jev bills no output tokens and has no
        `response_format`, so honouring either would be pretending.
        """
        import httpx

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise AxialError(
                "OPENROUTER_API_KEY is not set. Jev is served through OpenRouter's "
                "Decisions API, which needs its own key."
            )

        body = {
            "model": self.model,
            "state": prompt,
            "questions": {
                "verdict": {
                    "type": "noul",
                    "instructions": self.instructions,
                    "criteria": {"true": self.true_when, "false": self.false_when},
                }
            },
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        started = time.monotonic()
        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    DECISIONS_URL, json=body, headers=headers, timeout=self.timeout_s
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if terminal
                last_error = exc
                if attempt == self.max_retries:
                    break
                # Same jittered backoff as `LiteLLMModel`, for the same reason.
                time.sleep((2 ** (attempt - 1)) * (0.5 + self._rng.random()))
                continue

            served = payload.get("model") or self.model
            if not _served_model_matches(self.model, served):
                raise ServedModelMismatch(
                    f"requested {self.model!r} but the provider served {served!r}"
                )

            probability = self._read_noul(payload)
            failed = probability >= self.threshold
            reason = (
                f"jev noul={probability:.3f} threshold={self.threshold:g} "
                f"-> {'fail' if failed else 'pass'}"
            )
            return Completion(
                text=json.dumps({"reason": reason, "verdict": "fail" if failed else "pass"}),
                model=served,
                usage=_read_usage(payload.get("usage")),
                stop_reason="stop",
                truncated=False,
                attempts=attempt,
                latency_s=time.monotonic() - started,
                probability=probability,
            )

        raise AxialError(
            f"jev decision failed after {self.max_retries} attempts: {last_error}"
        )

    def _read_noul(self, payload: dict[str, Any]) -> float:
        answers = payload.get("answers") or {}
        answer = answers.get("verdict")
        if answer is None:
            raise AxialError(
                f"the decision response carried no 'verdict' answer: {sorted(answers)!r}"
            )
        if answer.get("type") != "noul":
            raise AxialError(
                f"expected a 'noul' answer but the provider returned {answer.get('type')!r}. "
                "A probability is the only answer shape this adapter can threshold."
            )
        value = answer.get("noul")
        if not isinstance(value, (int, float)):
            raise AxialError(f"the 'noul' answer carried no probability: {answer!r}")
        # Thresholded, anything else is a verdict nobody gave, and recorded it breaks
        # every Brier score and tuning run that reads it later.
        if not is_probability(value):
            raise AxialError(
                f"the 'noul' answer {value!r} is not a probability between 0 and 1"
            )
        return float(value)


def _read_usage(usage: object) -> dict[str, int]:
    """Read token counts, recording an absent count as absent.

    Invariant 9 again: no silent zeros. OpenRouter normalises to the OpenAI names,
    but the Decisions API is alpha, so the TypeSafe-native names are accepted too.
    """
    if not isinstance(usage, dict):
        return {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if input_tokens is None and output_tokens is None:
        return {}
    return {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}
