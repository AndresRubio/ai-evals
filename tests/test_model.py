import sys
import time
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from axial.errors import AxialError
from axial.model import Completion, FakeModel, LiteLLMModel, ServedModelMismatch


def test_fake_model_returns_a_mapped_response():
    m = FakeModel({"hello": "world"})
    out = m.complete("hello")
    assert isinstance(out, Completion)
    assert out.text == "world"
    assert out.model == "fake"


def test_fake_model_accepts_a_callable():
    m = FakeModel(lambda prompt: prompt.upper())
    assert m.complete("abc").text == "ABC"


def test_fake_model_is_deterministic():
    m = FakeModel(lambda prompt: f"{len(prompt)}")
    assert m.complete("abcd").text == m.complete("abcd").text


def test_fake_model_raises_for_an_unmapped_prompt():
    with pytest.raises(KeyError):
        FakeModel({"a": "b"}).complete("c")


def test_fake_model_records_usage_and_attempts():
    out = FakeModel({"a": "b"}).complete("a")
    assert out.usage["input_tokens"] > 0
    assert out.attempts == 1
    assert out.truncated is False


def test_fake_model_can_simulate_transient_failures_then_succeed():
    m = FakeModel({"a": "b"}, fail_times=2)
    out = m.complete("a")
    assert out.text == "b"
    assert out.attempts == 3


def test_served_model_mismatch_is_its_own_error():
    assert issubclass(ServedModelMismatch, Exception)


def test_fake_model_records_a_requested_schema():
    class Answer(BaseModel):
        verdict: str

    m = FakeModel({"a": "b"})
    m.complete("a", schema=Answer)
    assert m.schemas == [Answer]


def test_fake_model_ignores_the_schema_when_none_is_asked_for():
    m = FakeModel({"a": "b"})
    m.complete("a")
    assert m.schemas == [None]


def test_fake_model_can_simulate_truncation():
    m = FakeModel({"a": "b"}, truncated=True)
    out = m.complete("a")
    assert out.truncated is True
    assert out.stop_reason == "length"


def test_completion_default_stop_reason_is_stop():
    c = Completion(text="x", model="m")
    assert c.stop_reason == "stop"


def test_litellm_model_default_seed_is_none():
    """Production instances must not share a deterministic backoff sequence."""
    import inspect

    sig = inspect.signature(LiteLLMModel.__init__)
    assert sig.parameters["seed"].default is None


# --- LiteLLMModel, exercised against a stub `litellm` module (no network) ---


def _make_response(
    model="my-model",
    content="hello",
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=5,
):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _make_stub_litellm(completion_fn, supports_response_schema_fn=None):
    if supports_response_schema_fn is None:
        def supports_response_schema_fn(**kwargs):
            return True

    return SimpleNamespace(
        completion=completion_fn,
        supports_response_schema=supports_response_schema_fn,
    )


def test_litellm_model_successful_call_returns_completion(monkeypatch):
    response = _make_response(model="my-model", content="hello", prompt_tokens=10, completion_tokens=3)
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setitem(sys.modules, "litellm", _make_stub_litellm(fake_completion))

    model = LiteLLMModel("my-model", seed=1)
    out = model.complete("hi")

    assert out.text == "hello"
    assert out.model == "my-model"
    assert out.usage == {"input_tokens": 10, "output_tokens": 3}
    assert out.attempts == 1
    assert len(calls) == 1


def test_litellm_model_retries_transient_failure_then_succeeds(monkeypatch):
    response = _make_response(content="ok")
    seen = {"n": 0}

    def fake_completion(**kwargs):
        seen["n"] += 1
        if seen["n"] == 1:
            raise RuntimeError("rate limited")
        return response

    monkeypatch.setitem(sys.modules, "litellm", _make_stub_litellm(fake_completion))
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    model = LiteLLMModel("my-model", max_retries=3, seed=1)
    out = model.complete("hi")

    assert out.attempts == 2
    assert len(sleep_calls) == 1


def test_litellm_model_exhausts_retries_raises_axial_error(monkeypatch):
    def fake_completion(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "litellm", _make_stub_litellm(fake_completion))
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    model = LiteLLMModel("my-model", max_retries=3, seed=1)
    with pytest.raises(AxialError, match="boom"):
        model.complete("hi")

    assert len(sleep_calls) == 2  # max_retries - 1


def test_litellm_model_seed_is_passed_through_to_random(monkeypatch):
    def fake_completion(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "litellm", _make_stub_litellm(fake_completion))

    model_a = LiteLLMModel("my-model", max_retries=3, seed=42)
    model_b = LiteLLMModel("my-model", max_retries=3, seed=42)

    sleeps_a: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps_a.append(s))
    with pytest.raises(AxialError):
        model_a.complete("hi")

    sleeps_b: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps_b.append(s))
    with pytest.raises(AxialError):
        model_b.complete("hi")

    assert sleeps_a == sleeps_b


def test_litellm_model_truncated_when_finish_reason_is_length(monkeypatch):
    response = _make_response(finish_reason="length")
    monkeypatch.setitem(sys.modules, "litellm", _make_stub_litellm(lambda **kw: response))

    model = LiteLLMModel("my-model", seed=1)
    out = model.complete("hi")

    assert out.stop_reason == "length"
    assert out.truncated is True


def test_litellm_model_sets_response_format_when_schema_supported(monkeypatch):
    class Answer(BaseModel):
        verdict: str

    response = _make_response()
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return response

    stub = _make_stub_litellm(fake_completion, supports_response_schema_fn=lambda **kw: True)
    monkeypatch.setitem(sys.modules, "litellm", stub)

    model = LiteLLMModel("my-model", seed=1)
    model.complete("hi", schema=Answer)

    assert calls[0]["response_format"] is Answer


def test_litellm_model_omits_response_format_when_schema_unsupported(monkeypatch):
    class Answer(BaseModel):
        verdict: str

    response = _make_response()
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return response

    stub = _make_stub_litellm(fake_completion, supports_response_schema_fn=lambda **kw: False)
    monkeypatch.setitem(sys.modules, "litellm", stub)

    model = LiteLLMModel("my-model", seed=1)
    model.complete("hi", schema=Answer)

    assert "response_format" not in calls[0]


def test_litellm_model_falls_back_when_supports_response_schema_raises(monkeypatch):
    class Answer(BaseModel):
        verdict: str

    response = _make_response()
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return response

    def raising_supports(**kwargs):
        raise AttributeError("no such helper in this old litellm")

    stub = _make_stub_litellm(fake_completion, supports_response_schema_fn=raising_supports)
    monkeypatch.setitem(sys.modules, "litellm", stub)

    model = LiteLLMModel("my-model", seed=1)
    out = model.complete("hi", schema=Answer)

    assert "response_format" not in calls[0]
    assert out.text == "hello"


@pytest.mark.parametrize(
    "requested, served",
    [
        ("claude-opus-5", "claude-opus-5"),
        ("claude-opus-5", "anthropic/claude-opus-5"),
        ("claude-opus-5", "claude-opus-5-20260401"),
    ],
)
def test_litellm_model_accepts_matching_served_models(monkeypatch, requested, served):
    response = _make_response(model=served)
    monkeypatch.setitem(sys.modules, "litellm", _make_stub_litellm(lambda **kw: response))

    model = LiteLLMModel(requested, seed=1)
    out = model.complete("hi")

    assert out.model == served


@pytest.mark.parametrize(
    "requested, served",
    [
        ("gpt-4", "gpt-4o"),
        ("claude-opus-5", "claude-opus-5-legacy"),
    ],
)
def test_litellm_model_rejects_mismatched_served_models(monkeypatch, requested, served):
    response = _make_response(model=served)
    monkeypatch.setitem(sys.modules, "litellm", _make_stub_litellm(lambda **kw: response))

    model = LiteLLMModel(requested, seed=1)
    with pytest.raises(ServedModelMismatch):
        model.complete("hi")
