"""A judge model whose answer takes a set number of tokens, for the answer-cap tests."""

from __future__ import annotations

from collections.abc import Callable

from axial.core.model import Completion

REASON_FIRST_FAIL = '{"reason": "both stories share one headline", "verdict": "fail"}'


class LongAnswer:
    """Answers with `reply(prompt)`, which takes `needs` tokens with the model's thinking,
    and is cut off the way a provider cuts it when the cap is smaller. Records the cap
    each call was given."""

    def __init__(
        self,
        needs: int,
        reply: Callable[[str], str] = lambda _p: REASON_FIRST_FAIL,
        *,
        model: str = "fake",
    ) -> None:
        self.needs = needs
        self.reply = reply
        self.model = model
        self.caps: list[int] = []

    def complete(self, prompt: str, *, max_tokens: int = 1024, schema: object = None) -> Completion:
        self.caps.append(max_tokens)
        cut = max_tokens < self.needs
        return Completion(
            text=self.reply(prompt),
            model=self.model,
            stop_reason="length" if cut else "stop",
            truncated=cut,
        )
