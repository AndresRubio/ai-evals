"""Tests for scripts/knowledge_index.py, imported by path because it is not a package."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "knowledge_index.py"
_spec = importlib.util.spec_from_file_location("knowledge_index", SCRIPT)
assert _spec is not None and _spec.loader is not None
ki = importlib.util.module_from_spec(_spec)
sys.modules["knowledge_index"] = ki
_spec.loader.exec_module(ki)

GOOD = """---
name: evidently
description: Open-source ML and LLM evaluation and monitoring; tests, LLM judges, tracing, scheduled evals
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://docs.evidentlyai.com/llms.txt
    read: 2026-10-07
license: Apache-2.0
related: [landscape-overview]
---
# Evidently

Body.
"""


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_parse_entry_accepts_the_spec_example(tmp_path: Path) -> None:
    path = write(tmp_path, "knowledge/landscape/evidently.md", GOOD)
    entry = ki.parse_entry(path, tmp_path)
    assert entry.name == "evidently"
    assert entry.shelf == "landscape"
    assert entry.updated == date(2026, 10, 7)
    assert entry.sources == (ki.Source("https://docs.evidentlyai.com/llms.txt", date(2026, 10, 7)),)
    assert entry.related == ("landscape-overview",)
    assert entry.extra is None
    assert entry.license == "Apache-2.0"
    assert entry.body.startswith("# Evidently")


def violations(root: Path, rel: str, text: str) -> list[str]:
    path = write(root, rel, text)
    with pytest.raises(ki.KnowledgeError) as info:
        ki.parse_entry(path, root)
    return info.value.violations


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (lambda s: s.replace("updated: 2026-10-07\n", ""), "updated: missing"),
        (lambda s: s.replace("name: evidently", "name: other"), "name: 'other' does not match filename stem 'evidently'"),
        (lambda s: s.replace("name: evidently", "name: Evidently"), "name: 'Evidently' is not kebab-case"),
        (lambda s: s.replace("shelf: landscape", "shelf: procedures"), "shelf: 'procedures' does not match directory 'landscape'"),
        (lambda s: s.replace("shelf: landscape", "shelf: notes"), "shelf: 'notes' is not one of methodology, procedures, landscape"),
        (lambda s: s.replace("scheduled evals\n", "scheduled evals.\n"), "description: ends with a period"),
        (lambda s: s.replace("description: Open", "description: |\n  Open\n  two lines\n  "), "description: must be one line"),
        (lambda s: s.replace("description: Open-source", "description: " + "x" * 161 + " Open-source"), "description: 255 characters, limit 160"),
        (lambda s: s.replace("updated: 2026-10-07", "updated: '2026-10-07'"), "updated: not an ISO date"),
        (lambda s: s.replace("updated: 2026-10-07", "updated: 7 Oct 2026"), "updated: not an ISO date"),
        (lambda s: s.replace("sources:\n  - url: https://docs.evidentlyai.com/llms.txt\n    read: 2026-10-07\n", "sources: []\n"), "sources: must list at least one source"),
        (lambda s: s.replace("    read: 2026-10-07\n", ""), "sources[0].read: missing"),
        (lambda s: s.replace("license: Apache-2.0\n", "license: 3\n"), "license: must be a string"),
        (lambda s: s.replace("related: [landscape-overview]\n", "related: landscape-overview\n"), "related: must be a list of names"),
    ],
)
def test_parse_entry_rejects_schema_violations(tmp_path: Path, mutation, fragment: str) -> None:
    found = violations(tmp_path, "knowledge/landscape/evidently.md", mutation(GOOD))
    assert any(fragment in v for v in found), found


def test_parse_entry_reports_every_violation_in_one_file(tmp_path: Path) -> None:
    text = GOOD.replace("updated: 2026-10-07\n", "").replace("name: evidently", "name: other")
    found = violations(tmp_path, "knowledge/landscape/evidently.md", text)
    assert len(found) == 2


def test_parse_entry_rejects_missing_fence_and_bad_yaml(tmp_path: Path) -> None:
    assert violations(tmp_path, "knowledge/landscape/a.md", "# no frontmatter\n") == [
        "knowledge/landscape/a.md: no frontmatter block"
    ]
    assert violations(tmp_path, "knowledge/landscape/b.md", "---\nname: [\n---\nbody\n")[0].startswith(
        "knowledge/landscape/b.md: frontmatter is not valid YAML"
    )


def test_violation_messages_are_prefixed_with_the_relative_path(tmp_path: Path) -> None:
    found = violations(tmp_path, "knowledge/landscape/evidently.md", GOOD.replace("updated: 2026-10-07\n", ""))
    assert found == ["knowledge/landscape/evidently.md: updated: missing"]
