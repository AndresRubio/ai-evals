import pytest
from pydantic import BaseModel

from axial.model import Completion, FakeModel, ServedModelMismatch


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
