"""The one place that knows how to call a model.

Everything else takes a `Model`. That is what lets every test in this project run
offline against `FakeModel`, and what keeps retry, cost accounting and the
served-model check in a single place instead of scattered through the pipeline.
"""

from __future__ import annotations

import random
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
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "end_turn"
    truncated: bool = False
    attempts: int = 1
    latency_s: float = 0.0


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
    ) -> None:
        self._responses = responses
        self._model = model
        self._fail_times = fail_times
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
        seed: int = 0,
    ) -> None:
        self.model = model
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
            if not served.startswith(self.model.split("/")[-1]) and served != self.model:
                raise ServedModelMismatch(
                    f"requested {self.model!r} but the provider served {served!r}"
                )

            choice = response.choices[0]
            stop_reason = getattr(choice, "finish_reason", "stop") or "stop"
            usage = getattr(response, "usage", None)
            return Completion(
                text=choice.message.content or "",
                model=served,
                usage={
                    "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                },
                stop_reason=stop_reason,
                truncated=stop_reason == "length",
                attempts=attempt,
                latency_s=time.monotonic() - started,
            )

        raise AxialError(f"model call failed after {self.max_retries} attempts: {last_error}")
