from pathlib import Path

from fastapi.testclient import TestClient

from axial.annotate.app import create_app
from axial.sample import create_session, read_session
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
