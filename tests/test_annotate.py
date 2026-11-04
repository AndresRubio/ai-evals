from pathlib import Path

from fastapi.testclient import TestClient

from axial.annotate.app import create_app
from axial.reviews import create_session, read_session
from axial.store import Store
from axial.traces import Trace


def build(tmp_path: Path) -> tuple[Store, TestClient]:
    store = Store(tmp_path)
    traces = [Trace(trace_id=f"t{i}", input=f"in{i}", output=f"out{i}") for i in range(3)]
    store.write_traces(traces)
    create_session(store, traces, name="s1")
    return store, TestClient(create_app(store, "s1", "andres"))


def test_index_serves_the_page(tmp_path: Path):
    _, client = build(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "axial" in response.text.lower()


def test_session_endpoint_returns_traces_and_progress(tmp_path: Path):
    _, client = build(tmp_path)
    payload = client.get("/api/session").json()
    assert payload["total"] == 3
    assert payload["done"] == 0
    assert payload["items"][0]["input"] == "in0"


def test_posting_a_review_persists_it(tmp_path: Path):
    store, client = build(tmp_path)
    response = client.post(
        "/api/review/t1",
        json={"open_code": "merged two stories", "verdict": "fail", "critique": "both in one"},
    )
    assert response.status_code == 200
    saved = {r.trace_id: r for r in read_session(store, "s1")}["t1"]
    assert saved.open_code == "merged two stories"
    assert saved.verdict == "fail"
    assert saved.annotator == "andres"
    assert saved.status == "done"


def test_progress_reflects_completed_reviews(tmp_path: Path):
    _, client = build(tmp_path)
    client.post("/api/review/t0", json={"open_code": "x", "verdict": "pass", "critique": "fine"})
    assert client.get("/api/session").json()["done"] == 1


def test_posting_an_unknown_trace_is_a_404(tmp_path: Path):
    _, client = build(tmp_path)
    response = client.post("/api/review/nope", json={"open_code": "x", "verdict": "pass"})
    assert response.status_code == 404


def test_an_invalid_verdict_is_rejected(tmp_path: Path):
    _, client = build(tmp_path)
    response = client.post("/api/review/t0", json={"open_code": "x", "verdict": "maybe"})
    assert response.status_code == 422


def test_an_omitted_verdict_is_rejected(tmp_path: Path):
    _, client = build(tmp_path)
    response = client.post("/api/review/t0", json={"open_code": "x"})
    assert response.status_code == 422


def test_posting_an_unknown_trace_writes_nothing(tmp_path: Path):
    store, client = build(tmp_path)
    before = read_session(store, "s1")
    response = client.post("/api/review/nope", json={"open_code": "x", "verdict": "pass"})
    assert response.status_code == 404
    assert read_session(store, "s1") == before


def test_annotator_cannot_be_set_by_the_client(tmp_path: Path):
    store, client = build(tmp_path)
    response = client.post(
        "/api/review/t0",
        json={"open_code": "x", "verdict": "pass", "critique": "fine", "annotator": "someone-else"},
    )
    assert response.status_code == 200
    saved = {r.trace_id: r for r in read_session(store, "s1")}["t0"]
    assert saved.annotator == "andres"


def test_session_marks_an_item_whose_trace_is_missing_from_the_store(tmp_path: Path):
    store = Store(tmp_path)
    traces = [Trace(trace_id=f"t{i}", input=f"in{i}", output=f"out{i}") for i in range(3)]
    store.write_traces(traces)
    create_session(store, traces, name="s1")
    # Drop t1's trace from the store after the session was created, so its review
    # can no longer be resolved against a trace.
    store.write_traces([t for t in traces if t.trace_id != "t1"])
    client = TestClient(create_app(store, "s1", "andres"))

    payload = client.get("/api/session").json()

    by_id = {item["trace_id"]: item for item in payload["items"]}
    assert by_id["t1"]["missing"] is True
    assert by_id["t1"]["input"] == ""
    assert by_id["t1"]["output"] == ""
    assert by_id["t0"]["missing"] is False
    assert by_id["t2"]["missing"] is False


def test_keyboard_chords_and_hint_are_in_the_served_html(tmp_path: Path):
    _, client = build(tmp_path)
    html = client.get("/").text

    # Both chords must be documented in the visible hint...
    assert "⌘/Ctrl+Enter" in html
    assert "⌘/Ctrl+Shift+Enter" in html
    # ...and both verdicts must actually be reachable from the in-field handler.
    assert 'e.key === "Enter" && (e.metaKey || e.ctrlKey) && e.shiftKey' in html
    assert 'save("pass")' in html
    assert 'save("fail")' in html
