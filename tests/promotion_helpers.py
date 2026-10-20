"""For tests that promote a run they wrote by hand instead of through `run_judge`."""

from __future__ import annotations

from axial.judge import JudgeSpec, current_prompt_sha256
from axial.store import Store


def stamp_run(store: Store, mode: str, version: int) -> None:
    """Write a default judge.yaml if there is none, and stamp every verdict and error row
    with that judge's prompt fingerprint, as `run_judge` would have."""
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "judge.yaml").exists():
        JudgeSpec(mode=mode, model="fake", version=version).save(store)
    fingerprint = current_prompt_sha256(store, JudgeSpec.load(store, mode, version))
    for name in ("verdicts.jsonl", "errors.jsonl"):
        path = directory / name
        if path.exists():
            store.write_jsonl(
                path, [{**row, "prompt_sha256": fingerprint} for row in store.read_jsonl(path)]
            )
