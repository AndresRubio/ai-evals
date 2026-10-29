"""`JevModel`: the OpenRouter Decisions API behind the `Model` protocol.

Every test here runs offline against a stub `httpx` module, the same way
`test_model.py` stubs `litellm`. Nothing in this file touches the network.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from axial.errors import AxialError
from axial.model import JevModel, ServedModelMismatch

PINNED = "typesafe/jev-1.13"


def _make_response(
    *,
    noul: float = 0.9,
    model: str = PINNED,
    status: int = 200,
    usage: dict[str, int] | None = None,
    answers: dict[str, object] | None = None,
):
    if answers is None:
        answers = {"verdict": {"type": "noul", "noul": noul}}
    payload: dict[str, object] = {"model": model, "answers": answers}
    if usage is not None:
        payload["usage"] = usage

    return SimpleNamespace(
        status_code=status,
        json=lambda: payload,
        raise_for_status=lambda: None,
        text=json.dumps(payload),
    )


def _stub_httpx(post_fn):
    class _HTTPError(Exception):
        pass

    return SimpleNamespace(post=post_fn, HTTPError=_HTTPError, TimeoutException=_HTTPError)


def _install(monkeypatch, post_fn, *, api_key: str = "sk-or-test") -> None:
    monkeypatch.setitem(sys.modules, "httpx", _stub_httpx(post_fn))
    monkeypatch.setenv("OPENROUTER_API_KEY", api_key)


def _model(**kwargs) -> JevModel:
    kwargs.setdefault("model", PINNED)
    kwargs.setdefault("instructions", "Does this trace exhibit the failure mode?")
    kwargs.setdefault("true_when", "the failure mode is present")
    kwargs.setdefault("false_when", "the failure mode is absent")
    kwargs.setdefault("seed", 1)
    return JevModel(**kwargs)


# --- the verdict it synthesises ---------------------------------------------


def test_probability_at_or_above_threshold_is_a_fail_verdict(monkeypatch):
    _install(monkeypatch, lambda *a, **kw: _make_response(noul=0.80))

    out = _model(threshold=0.80).complete("grade this trace")

    assert json.loads(out.text)["verdict"] == "fail"


def test_probability_below_threshold_is_a_pass_verdict(monkeypatch):
    _install(monkeypatch, lambda *a, **kw: _make_response(noul=0.79))

    out = _model(threshold=0.80).complete("grade this trace")

    assert json.loads(out.text)["verdict"] == "pass"


def test_reason_records_the_probability_and_the_threshold(monkeypatch):
    """Jev returns no critique. The probability is the only audit trail there is."""
    _install(monkeypatch, lambda *a, **kw: _make_response(noul=0.912))

    out = _model(threshold=0.5).complete("grade this trace")

    reason = json.loads(out.text)["reason"]
    assert "0.912" in reason
    assert "0.5" in reason


def test_probability_is_exposed_on_the_completion(monkeypatch):
    """A threshold sweep needs the raw probability, not the thresholded verdict."""
    _install(monkeypatch, lambda *a, **kw: _make_response(noul=0.4321))

    out = _model(threshold=0.5).complete("grade this trace")

    assert out.probability == pytest.approx(0.4321)


# --- the request it sends ----------------------------------------------------


def test_sends_the_prompt_as_state_with_one_noul_question(monkeypatch):
    calls: list[dict] = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _make_response()

    _install(monkeypatch, post)
    _model(
        instructions="Is the answer unsupported by the context?",
        true_when="unsupported",
        false_when="supported",
    ).complete("the assembled judge prompt")

    body = calls[0]["json"]
    assert body["model"] == PINNED
    assert body["state"] == "the assembled judge prompt"
    assert list(body["questions"]) == ["verdict"]
    question = body["questions"]["verdict"]
    assert question["type"] == "noul"
    assert question["instructions"] == "Is the answer unsupported by the context?"
    assert question["criteria"] == {"true": "unsupported", "false": "supported"}


def test_posts_to_the_decisions_endpoint_with_the_api_key(monkeypatch):
    calls: list[dict] = []

    def post(*args, **kwargs):
        calls.append({"args": args, **kwargs})
        return _make_response()

    _install(monkeypatch, post, api_key="sk-or-v1-secret")
    _model().complete("x")

    assert calls[0]["args"][0] == "https://openrouter.ai/api/alpha/decisions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-or-v1-secret"


def test_missing_api_key_is_refused_before_any_request(monkeypatch):
    calls: list[dict] = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _make_response()

    monkeypatch.setitem(sys.modules, "httpx", _stub_httpx(post))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(AxialError, match="OPENROUTER_API_KEY"):
        _model().complete("x")
    assert calls == []


# --- the pin ----------------------------------------------------------------


def test_a_floating_model_alias_is_refused(monkeypatch):
    """A moving alias swaps the judge under a PROMOTED marker whose hashes still match."""
    with pytest.raises(AxialError, match="pinned"):
        _model(model="~typesafe/jev-latest")


def test_a_pinned_version_is_accepted(monkeypatch):
    assert _model(model="typesafe/jev-1.13").model == "typesafe/jev-1.13"


def test_serving_a_different_model_raises(monkeypatch):
    _install(monkeypatch, lambda *a, **kw: _make_response(model="typesafe/jev-2.0"))

    with pytest.raises(ServedModelMismatch):
        _model().complete("x")


# --- honesty about what it does and does not report --------------------------


def test_usage_is_empty_when_the_response_reports_none(monkeypatch):
    """Invariant 9: an unknown cost is recorded as unknown, never as a silent zero."""
    _install(monkeypatch, lambda *a, **kw: _make_response(usage=None))

    assert _model().complete("x").usage == {}


def test_usage_is_read_when_the_response_reports_it(monkeypatch):
    _install(
        monkeypatch,
        lambda *a, **kw: _make_response(usage={"prompt_tokens": 1200, "completion_tokens": 0}),
    )

    assert _model().complete("x").usage == {"input_tokens": 1200, "output_tokens": 0}


def test_a_decision_is_never_truncated(monkeypatch):
    """Jev emits no output tokens, so `length` truncation cannot happen to it."""
    _install(monkeypatch, lambda *a, **kw: _make_response())

    out = _model().complete("x")

    assert out.truncated is False
    assert out.stop_reason == "stop"


def test_a_response_schema_is_never_sent(monkeypatch):
    """The Decisions API has no `response_format`; asking for one would be a lie."""
    from pydantic import BaseModel

    class Anything(BaseModel):
        verdict: str

    calls: list[dict] = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _make_response()

    _install(monkeypatch, post)
    _model().complete("x", schema=Anything)

    assert "response_format" not in calls[0]["json"]


# --- failures ----------------------------------------------------------------


def test_an_answer_of_the_wrong_type_is_an_error_not_a_guess(monkeypatch):
    _install(
        monkeypatch,
        lambda *a, **kw: _make_response(answers={"verdict": {"type": "choice", "choice": "billing"}}),
    )

    with pytest.raises(AxialError, match="noul"):
        _model().complete("x")


@pytest.mark.parametrize("bad", [float("nan"), 1.7, -0.2, True])
def test_a_noul_that_is_not_a_probability_is_an_error(monkeypatch, bad):
    # Recorded as it came, a value like this would crash `align` and `promote` later
    # and be thresholded into a verdict now (nan >= t is False: a silent pass).
    _install(monkeypatch, lambda *a, **kw: _make_response(noul=bad))

    with pytest.raises(AxialError, match="between 0 and 1"):
        _model().complete("x")


def test_a_missing_answer_is_an_error(monkeypatch):
    _install(monkeypatch, lambda *a, **kw: _make_response(answers={}))

    with pytest.raises(AxialError):
        _model().complete("x")


def test_a_transport_failure_is_retried_then_succeeds(monkeypatch):
    attempts: list[int] = []

    def post(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise sys.modules["httpx"].HTTPError("boom")
        return _make_response(noul=0.99)

    _install(monkeypatch, post)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    out = _model(threshold=0.5, max_retries=3).complete("x")

    assert json.loads(out.text)["verdict"] == "fail"
    assert out.attempts == 3


def test_exhausting_the_retries_raises(monkeypatch):
    def post(*args, **kwargs):
        raise sys.modules["httpx"].HTTPError("boom")

    _install(monkeypatch, post)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    with pytest.raises(AxialError, match="3 attempts"):
        _model(max_retries=3).complete("x")


# --- wiring: which model a judge spec gets -----------------------------------


def test_a_typesafe_id_is_recognised_as_a_decision_model():
    from axial.model import is_decision_model

    assert is_decision_model("typesafe/jev-1.13") is True
    assert is_decision_model("openrouter/typesafe/jev-1.13") is True
    assert is_decision_model("~typesafe/jev-latest") is True


def test_a_text_model_id_is_not_a_decision_model():
    from axial.model import is_decision_model

    assert is_decision_model("claude-opus-5") is False
    assert is_decision_model("anthropic/claude-opus-5") is False


def test_a_decision_model_is_refused_where_prose_is_required():
    """Taxonomy proposal needs generated text. Jev cannot write a sentence."""
    from axial.model import require_text_model

    with pytest.raises(AxialError, match="cannot generate text"):
        require_text_model("typesafe/jev-1.13")


def test_the_judge_spec_threshold_is_written_into_judge_yaml(tmp_path):
    """The threshold changes every verdict, so judge.yaml's hash must cover it."""
    import yaml

    from axial.judge import JudgeSpec
    from axial.store import Store

    store = Store(str(tmp_path))
    spec = JudgeSpec(mode="unsupported", model=PINNED, decision_threshold=0.8)
    spec.save(store)

    saved = yaml.safe_load((store.judge_dir("unsupported", 1) / "judge.yaml").read_text())
    assert saved["decision_threshold"] == 0.8


def test_a_spec_naming_jev_builds_a_jev_model_carrying_its_threshold():
    from axial.judge import JudgeSpec, build_judge_model

    spec = JudgeSpec(
        mode="unsupported",
        model=PINNED,
        mode_name="unsupported claim",
        mode_definition="the answer asserts something the context does not support",
        decision_threshold=0.75,
    )

    model = build_judge_model(spec)

    assert isinstance(model, JevModel)
    assert model.model == PINNED
    assert model.threshold == 0.75
    assert "does not support" in model.instructions


def test_a_spec_naming_a_text_model_builds_a_litellm_model():
    from axial.judge import JudgeSpec, build_judge_model
    from axial.model import LiteLLMModel

    model = build_judge_model(JudgeSpec(mode="unsupported", model="claude-opus-5"))

    assert isinstance(model, LiteLLMModel)
    assert model.model == "claude-opus-5"


def test_a_different_major_version_with_the_same_suffix_is_caught(monkeypatch):
    """`jev-1.13` and `jev-2.13` must not collide.

    The served-model check normalises a provider prefix away by splitting on the
    last `.` — which, for a dotted version, throws away everything but the minor
    number and makes two different models compare equal.
    """
    _install(monkeypatch, lambda *a, **kw: _make_response(model="typesafe/jev-2.13"))

    with pytest.raises(ServedModelMismatch):
        _model(model="typesafe/jev-1.13").complete("x")
