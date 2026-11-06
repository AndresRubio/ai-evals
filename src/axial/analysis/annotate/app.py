"""The annotation app.

Annotation throughput is the real bottleneck in error analysis, so this is a real
page with keyboard shortcuts rather than a terminal prompt — long multi-turn traces
and JSON payloads read badly in a terminal. It is one static file and one FastAPI
app; there is no build step and nothing to install from npm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from axial.core.reviews import read_session, write_session
from axial.core.store import Store

STATIC = Path(__file__).parent / "static"


class ReviewIn(BaseModel):
    open_code: str = ""
    verdict: Literal["pass", "fail"]
    critique: str = ""


def create_app(store: Store, session: str, annotator: str) -> FastAPI:
    app = FastAPI(title="axial annotate")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/session")
    def get_session() -> dict[str, Any]:
        reviews = read_session(store, session)
        traces = {t.trace_id: t for t in store.read_traces()}
        items = []
        for review in reviews:
            trace = traces.get(review.trace_id)
            items.append(
                {
                    "trace_id": review.trace_id,
                    "input": trace.input if trace else "",
                    "output": trace.output if trace else "",
                    "metadata": trace.metadata if trace else {},
                    "missing": trace is None,
                    "open_code": review.open_code,
                    "verdict": review.verdict,
                    "critique": review.critique,
                    "status": review.status,
                }
            )
        return {
            "session": session,
            "annotator": annotator,
            "total": len(reviews),
            "done": sum(1 for r in reviews if r.status == "done"),
            "items": items,
        }

    # `:path`, because the server decodes the page's %2F before routing, and a plain
    # parameter stops at "/": an id such as "run/123" from an adapter would never match.
    @app.post("/api/review/{trace_id:path}")
    def post_review(trace_id: str, body: ReviewIn) -> dict[str, Any]:
        reviews = read_session(store, session)
        for review in reviews:
            if review.trace_id == trace_id:
                review.open_code = body.open_code
                review.verdict = body.verdict
                review.critique = body.critique
                review.annotator = annotator
                review.status = "done"
                write_session(store, session, reviews)
                return {"ok": True, "done": sum(1 for r in reviews if r.status == "done")}
        raise HTTPException(status_code=404, detail=f"no trace {trace_id!r} in this session")

    return app
