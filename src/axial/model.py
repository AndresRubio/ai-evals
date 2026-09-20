"""The one place that knows how to call a model.

Everything else takes a `Model`. That is what lets every test in this project run
offline against `FakeModel`, and what keeps retry, cost accounting and the
served-model check in a single place instead of scattered through the pipeline.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

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


_DATED_SNAPSHOT_SUFFIX = re.compile(r"^(?P<base>.+)-\d{8}$")


def _normalise_model_id(model_id: str) -> str:
    """Strip a provider/routing prefix from a model id.

    `anthropic/claude-opus-5` and `bedrock.claude-opus-5` both reduce to
    `claude-opus-5` by taking the segment after the last `/` and after the
    last `.`.
    """
    return model_id.rsplit("/", 1)[-1].rsplit(".", 1)[-1]


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
