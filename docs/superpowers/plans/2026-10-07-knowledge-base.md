# Knowledge base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `docs/research/` into a queryable knowledge base: a `knowledge/` tree of one-topic Markdown files with validated frontmatter, indexed by a generated `llms.txt` that CI keeps honest.

**Architecture:** A standalone script, `scripts/knowledge_index.py`, parses every file under `knowledge/<shelf>/`, validates the schema and cross-file rules from the spec, renders `llms.txt`, and in `--check` mode fails when the committed index is stale. The script ships nothing into the `axial` wheel. Content tasks migrate the two research docs into three shelves and add the entries and procedures the spec names.

**Tech Stack:** Python 3.14 (floor 3.13), uv, PyYAML (already a dependency), `tomllib` from the standard library, pytest, ruff.

**Spec:** [`docs/superpowers/specs/2026-10-07-knowledge-base-layout-design.md`](../specs/2026-10-07-knowledge-base-layout-design.md). Read it first; every rule below comes from it.

## Global Constraints

- Branch: `knowledge-base`, already created from `main`. All commits go there.
- Commit identity and trailer, on every commit:

  ```bash
  git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
  <message>

  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  EOF
  ```

- Shelves are exactly `methodology`, `procedures`, `landscape`. Nothing else may live under `knowledge/` except `README.md`.
- Frontmatter required keys: `name`, `description`, `shelf`, `updated`, `sources` (list of `{url, read}`). Optional: `related`, `extra`, `license`.
- `name` equals the filename stem, matches `^[a-z0-9]+(-[a-z0-9]+)*$`, unique across shelves. `shelf` equals the parent directory.
- `description`: one line, at most 160 characters, no trailing period.
- Dates are ISO `YYYY-MM-DD` and must parse as YAML dates, not strings.
- Every landscape file except `landscape-overview` has these seven headings in this order: `## What it is`, `## Core abstraction`, `## Where it sits`, `## Maturity`, `## Biggest complaint`, `## Relation to axial`, `## Corrections`. A heading with nothing to say holds one line, `Not applicable.` or `None.`
- The failure is the positive class wherever TPR and TNR appear.
- The script collects every violation before exiting and never writes `llms.txt` for a broken tree.
- `ruff check .` must pass; the script is under `scripts/`, which ruff covers. mypy covers only `src/axial` and is not run on the script, but write it typed anyway.
- Tests never touch the network.

---

### Task 1: Frontmatter parsing and the per-file schema

**Files:**
- Create: `scripts/knowledge_index.py`
- Test: `tests/test_knowledge_index.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `KnowledgeError(violations: list[str])`; `Source(url: str, read: date)`; `Entry(name, description, shelf, updated, sources, related, extra, license, path, body)`; `SHELVES`; `split_frontmatter(text: str) -> tuple[dict[str, Any], str]`; `parse_entry(path: Path, root: Path) -> Entry`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_index.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_knowledge_index.py -q`
Expected: collection error, `FileNotFoundError` for `scripts/knowledge_index.py`.

- [ ] **Step 3: Write the parser**

Create `scripts/knowledge_index.py`:

```python
"""Build and check the knowledge base index.

    python scripts/knowledge_index.py            # rewrite llms.txt
    python scripts/knowledge_index.py --check    # validate; exit 1 on any violation

The tree under knowledge/ is the source of truth. llms.txt is generated from each
file's frontmatter. Every violation is collected before exiting so one run shows the
whole list, and a broken tree never produces an index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

SHELVES: tuple[str, ...] = ("methodology", "procedures", "landscape")
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
DESCRIPTION_MAX = 160
INDEX_NAME = "llms.txt"
KNOWLEDGE_DIR = "knowledge"
REQUIRED_KEYS = ("name", "description", "shelf", "updated", "sources")
OPTIONAL_KEYS = ("related", "extra", "license")


class KnowledgeError(Exception):
    """One or more violations, each a line of the form `path: message`."""

    def __init__(self, violations: list[str]) -> None:
        super().__init__("\n".join(violations))
        self.violations = violations


@dataclass(frozen=True)
class Source:
    url: str
    read: date


@dataclass(frozen=True)
class Entry:
    name: str
    description: str
    shelf: str
    updated: date
    sources: tuple[Source, ...]
    related: tuple[str, ...]
    extra: str | None
    license: str | None
    path: Path
    body: str


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter, body). Raises ValueError or TypeError with a reason when malformed."""
    if not text.startswith("---\n"):
        raise ValueError("no frontmatter block")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("no frontmatter block")
    raw = text[4:end]
    body = text[end + len("\n---\n") :]
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError("frontmatter is not a mapping")
    return data, body


def _is_date(value: object) -> bool:
    return isinstance(value, date) and not isinstance(value, datetime)


def _check_sources(raw: object, out: list[str]) -> tuple[Source, ...]:
    if not isinstance(raw, list) or not raw:
        out.append("sources: must list at least one source")
        return ()
    sources: list[Source] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            out.append(f"sources[{i}]: must be a mapping with url and read")
            continue
        url = item.get("url")
        read = item.get("read")
        ok = True
        if not isinstance(url, str) or not url:
            out.append(f"sources[{i}].url: missing")
            ok = False
        if read is None:
            out.append(f"sources[{i}].read: missing")
            ok = False
        elif not _is_date(read):
            out.append(f"sources[{i}].read: not an ISO date")
            ok = False
        if ok:
            sources.append(Source(str(url), read))  # type: ignore[arg-type]
    return tuple(sources)


def parse_entry(path: Path, root: Path) -> Entry:
    """Parse one knowledge file. `root` is the repository root, used for messages."""
    rel = path.relative_to(root).as_posix()
    problems: list[str] = []
    try:
        data, body = split_frontmatter(path.read_text())
    except (ValueError, TypeError) as exc:
        raise KnowledgeError([f"{rel}: {exc}"]) from exc

    for key in REQUIRED_KEYS:
        if key not in data:
            problems.append(f"{key}: missing")
    for key in data:
        if key not in REQUIRED_KEYS and key not in OPTIONAL_KEYS:
            problems.append(f"{key}: unknown key")

    name = data.get("name")
    if isinstance(name, str):
        if not NAME_RE.fullmatch(name):
            problems.append(f"name: '{name}' is not kebab-case")
        elif name != path.stem:
            problems.append(f"name: '{name}' does not match filename stem '{path.stem}'")
    elif "name" in data:
        problems.append("name: must be a string")

    description = data.get("description")
    if isinstance(description, str):
        if "\n" in description.strip():
            problems.append("description: must be one line")
        else:
            description = description.strip()
            if not description:
                problems.append("description: empty")
            if description.endswith("."):
                problems.append("description: ends with a period")
            if len(description) > DESCRIPTION_MAX:
                problems.append(
                    f"description: {len(description)} characters, limit {DESCRIPTION_MAX}"
                )
    elif "description" in data:
        problems.append("description: must be a string")

    shelf = data.get("shelf")
    if isinstance(shelf, str):
        if shelf not in SHELVES:
            problems.append(f"shelf: '{shelf}' is not one of {', '.join(SHELVES)}")
        elif shelf != path.parent.name:
            problems.append(f"shelf: '{shelf}' does not match directory '{path.parent.name}'")
    elif "shelf" in data:
        problems.append("shelf: must be a string")

    updated = data.get("updated")
    if "updated" in data and not _is_date(updated):
        problems.append("updated: not an ISO date")

    sources = _check_sources(data["sources"], problems) if "sources" in data else ()

    related_raw = data.get("related", [])
    if not isinstance(related_raw, list) or not all(isinstance(r, str) for r in related_raw):
        problems.append("related: must be a list of names")
        related_raw = []

    extra = data.get("extra")
    if extra is not None and not isinstance(extra, str):
        problems.append("extra: must be a string")
        extra = None

    license_ = data.get("license")
    if license_ is not None and not isinstance(license_, str):
        problems.append("license: must be a string")
        license_ = None

    if problems:
        raise KnowledgeError([f"{rel}: {p}" for p in problems])

    return Entry(
        name=str(name),
        description=str(description).strip(),
        shelf=str(shelf),
        updated=updated,  # type: ignore[arg-type]
        sources=sources,
        related=tuple(related_raw),
        extra=extra,
        license=license_,
        path=path,
        body=body,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_knowledge_index.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check scripts tests/test_knowledge_index.py
git add scripts/knowledge_index.py tests/test_knowledge_index.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
Parse knowledge-base frontmatter and reject every schema violation at once

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
```

---

### Task 2: Walking the tree and the cross-file rules

**Files:**
- Modify: `scripts/knowledge_index.py`
- Test: `tests/test_knowledge_index.py`

**Interfaces:**
- Consumes: `parse_entry`, `Entry`, `KnowledgeError`, `SHELVES`, `KNOWLEDGE_DIR` from Task 1.
- Produces: `load_tree(root: Path) -> list[Entry]` where `root` is the repository root; reads `root / "knowledge"` and `root / "pyproject.toml"`. Entries are returned sorted by `(shelf order, name)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_index.py`:

```python
def entry_text(name: str, shelf: str, body: str = "Body.\n", **extra_keys: str) -> str:
    lines = [
        "---",
        f"name: {name}",
        f"description: About {name}",
        f"shelf: {shelf}",
        "updated: 2026-10-07",
        "sources:",
        "  - url: https://example.invalid/" + name,
        "    read: 2026-10-07",
    ]
    lines += [f"{k}: {v}" for k, v in extra_keys.items()]
    lines += ["---", f"# {name}", "", body]
    return "\n".join(lines)


def make_repo(tmp_path: Path, files: dict[str, str], pyproject: str = '[project]\nname = "x"\n') -> Path:
    write(tmp_path, "pyproject.toml", pyproject)
    write(tmp_path, "knowledge/README.md", "# rules\n")
    for rel, text in files.items():
        write(tmp_path, "knowledge/" + rel, text)
    return tmp_path


def test_load_tree_returns_entries_in_shelf_order_then_by_name(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {
            "landscape/zeta.md": entry_text("zeta", "landscape"),
            "landscape/alpha.md": entry_text("alpha", "landscape"),
            "methodology/spine.md": entry_text("spine", "methodology"),
            "procedures/loop.md": entry_text("loop", "procedures"),
        },
    )
    assert [e.name for e in ki.load_tree(root)] == ["spine", "loop", "alpha", "zeta"]


def tree_violations(root: Path) -> list[str]:
    with pytest.raises(ki.KnowledgeError) as info:
        ki.load_tree(root)
    return info.value.violations


def test_load_tree_rejects_duplicate_names_across_shelves(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {
            "landscape/same.md": entry_text("same", "landscape"),
            "procedures/same.md": entry_text("same", "procedures"),
        },
    )
    assert tree_violations(root) == [
        "knowledge/procedures/same.md: name: 'same' is also used by knowledge/landscape/same.md"
    ]


def test_load_tree_rejects_unknown_related_name(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"landscape/a.md": entry_text("a", "landscape", related="[ghost]")})
    assert tree_violations(root) == ["knowledge/landscape/a.md: related: no entry named 'ghost'"]


def test_load_tree_rejects_broken_relative_link_and_accepts_good_ones(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {
            "landscape/a.md": entry_text(
                "a",
                "landscape",
                body="See [b](../procedures/b.md), [b again](../procedures/b.md#top), "
                "[site](https://example.invalid/), [anchor](#what-it-is), "
                "[missing](../procedures/c.md).\n",
            ),
            "procedures/b.md": entry_text("b", "procedures"),
        },
    )
    assert tree_violations(root) == [
        "knowledge/landscape/a.md: link: '../procedures/c.md' does not exist"
    ]


def test_load_tree_rejects_extra_absent_from_pyproject(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {"landscape/a.md": entry_text("a", "landscape", extra="evidently")},
        pyproject='[project]\nname = "x"\n[project.optional-dependencies]\nother = ["y"]\n',
    )
    assert tree_violations(root) == [
        "knowledge/landscape/a.md: extra: 'evidently' is not in [project.optional-dependencies]"
    ]


def test_load_tree_accepts_extra_present_in_pyproject(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {"landscape/a.md": entry_text("a", "landscape", extra="evidently")},
        pyproject='[project]\nname = "x"\n[project.optional-dependencies]\nevidently = ["evidently"]\n',
    )
    assert ki.load_tree(root)[0].extra == "evidently"


def test_load_tree_rejects_extra_outside_landscape(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {"procedures/a.md": entry_text("a", "procedures", extra="evidently")},
        pyproject='[project]\nname = "x"\n[project.optional-dependencies]\nevidently = ["evidently"]\n',
    )
    assert tree_violations(root) == ["knowledge/procedures/a.md: extra: only landscape entries may declare it"]


def test_load_tree_rejects_stray_paths(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"landscape/a.md": entry_text("a", "landscape")})
    write(root, "knowledge/notes.md", "stray\n")
    write(root, "knowledge/landscape/data.json", "{}")
    write(root, "knowledge/extra/x.md", entry_text("x", "extra"))
    assert tree_violations(root) == [
        "knowledge/extra: not a shelf; only methodology, procedures, landscape are allowed",
        "knowledge/landscape/data.json: not a Markdown file",
        "knowledge/notes.md: stray file; only README.md and shelf directories belong here",
    ]


def test_load_tree_collects_violations_from_several_files(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {
            "landscape/a.md": entry_text("a", "landscape").replace("updated: 2026-10-07\n", ""),
            "landscape/b.md": entry_text("b", "landscape", related="[ghost]"),
        },
    )
    found = tree_violations(root)
    assert found == [
        "knowledge/landscape/a.md: updated: missing",
        "knowledge/landscape/b.md: related: no entry named 'ghost'",
    ]


def test_load_tree_fails_when_knowledge_dir_is_missing(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", '[project]\nname = "x"\n')
    assert tree_violations(tmp_path) == ["knowledge: directory does not exist"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_knowledge_index.py -q -k load_tree`
Expected: FAIL with `AttributeError: module 'knowledge_index' has no attribute 'load_tree'`.

- [ ] **Step 3: Implement `load_tree`**

Add to `scripts/knowledge_index.py`, after `parse_entry`. Also add `import tomllib` beside the other imports.

```python
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")


def _optional_extras(pyproject: Path) -> set[str]:
    if not pyproject.exists():
        return set()
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    project = data.get("project", {})
    return set(project.get("optional-dependencies", {}).keys())


def _layout_violations(knowledge: Path, root: Path) -> tuple[list[str], list[Path]]:
    """Return (violations, markdown files to parse), checking only the tree's shape."""
    problems: list[str] = []
    files: list[Path] = []
    for child in sorted(knowledge.iterdir()):
        rel = child.relative_to(root).as_posix()
        if child.is_dir():
            if child.name not in SHELVES:
                problems.append(f"{rel}: not a shelf; only {', '.join(SHELVES)} are allowed")
                continue
            for item in sorted(child.iterdir()):
                item_rel = item.relative_to(root).as_posix()
                if item.is_dir() or item.suffix != ".md":
                    problems.append(f"{item_rel}: not a Markdown file")
                else:
                    files.append(item)
        elif child.name != "README.md":
            problems.append(f"{rel}: stray file; only README.md and shelf directories belong here")
    return problems, files


def _link_violations(entry: Entry, root: Path) -> list[str]:
    rel = entry.path.relative_to(root).as_posix()
    problems: list[str] = []
    for target in LINK_RE.findall(entry.body):
        if target.startswith(EXTERNAL_PREFIXES) or target.startswith("#"):
            continue
        file_part = target.split("#", 1)[0]
        if not (entry.path.parent / file_part).exists():
            problems.append(f"{rel}: link: '{target}' does not exist")
    return problems


def load_tree(root: Path) -> list[Entry]:
    """Parse every file under root/knowledge and apply the cross-file rules.

    Raises KnowledgeError with every violation found; returns entries sorted by
    shelf order then name.
    """
    knowledge = root / KNOWLEDGE_DIR
    if not knowledge.is_dir():
        raise KnowledgeError([f"{KNOWLEDGE_DIR}: directory does not exist"])

    problems, files = _layout_violations(knowledge, root)
    entries: list[Entry] = []
    for path in files:
        try:
            entries.append(parse_entry(path, root))
        except KnowledgeError as exc:
            problems.extend(exc.violations)

    by_name: dict[str, Entry] = {}
    for entry in entries:
        rel = entry.path.relative_to(root).as_posix()
        if entry.name in by_name:
            other = by_name[entry.name].path.relative_to(root).as_posix()
            problems.append(f"{rel}: name: '{entry.name}' is also used by {other}")
        else:
            by_name[entry.name] = entry

    extras = _optional_extras(root / "pyproject.toml")
    for entry in entries:
        rel = entry.path.relative_to(root).as_posix()
        for name in entry.related:
            if name not in by_name:
                problems.append(f"{rel}: related: no entry named '{name}'")
        if entry.extra is not None:
            if entry.shelf != "landscape":
                problems.append(f"{rel}: extra: only landscape entries may declare it")
            elif entry.extra not in extras:
                problems.append(
                    f"{rel}: extra: '{entry.extra}' is not in [project.optional-dependencies]"
                )
        problems.extend(_link_violations(entry, root))

    if problems:
        raise KnowledgeError(sorted(problems))
    entries.sort(key=lambda e: (SHELVES.index(e.shelf), e.name))
    return entries
```

Note the final `sorted(problems)`: messages sort by path first, which is what the "collects violations from several files" and "stray paths" tests assert.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_knowledge_index.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check scripts tests/test_knowledge_index.py
git add scripts/knowledge_index.py tests/test_knowledge_index.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
Walk the knowledge tree and enforce its cross-file rules

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
```

---

### Task 3: Rendering `llms.txt`, the check, and the command line

**Files:**
- Modify: `scripts/knowledge_index.py`
- Test: `tests/test_knowledge_index.py`

**Interfaces:**
- Consumes: `load_tree`, `Entry`, `KnowledgeError`, `SHELVES`, `INDEX_NAME`.
- Produces: `render_index(entries: list[Entry]) -> str`; `check(root: Path) -> list[str]`; `write_index(root: Path) -> list[str]`; `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_index.py`:

```python
EXPECTED_INDEX = """# axial knowledge base

> Everything this repository has read about evaluating LLM applications, as one file
> per topic. Fetch a file by its path. Generated by scripts/knowledge_index.py; edit the
> files, not this index.

## methodology

- [spine](knowledge/methodology/spine.md): About spine (2026-10-07)

## procedures

- [loop](knowledge/procedures/loop.md): About loop (2026-10-07)

## landscape

- [alpha](knowledge/landscape/alpha.md): About alpha (2026-10-07)
- [zeta](knowledge/landscape/zeta.md): About zeta (2026-10-07)
"""


def three_shelf_repo(tmp_path: Path) -> Path:
    return make_repo(
        tmp_path,
        {
            "landscape/zeta.md": entry_text("zeta", "landscape"),
            "landscape/alpha.md": entry_text("alpha", "landscape"),
            "methodology/spine.md": entry_text("spine", "methodology"),
            "procedures/loop.md": entry_text("loop", "procedures"),
        },
    )


def test_render_index_matches_the_spec_format(tmp_path: Path) -> None:
    root = three_shelf_repo(tmp_path)
    assert ki.render_index(ki.load_tree(root)) == EXPECTED_INDEX


def test_render_index_omits_empty_shelves(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"landscape/alpha.md": entry_text("alpha", "landscape")})
    text = ki.render_index(ki.load_tree(root))
    assert "## methodology" not in text and "## procedures" not in text
    assert "## landscape" in text


def test_check_passes_when_index_matches(tmp_path: Path) -> None:
    root = three_shelf_repo(tmp_path)
    write(root, "llms.txt", EXPECTED_INDEX)
    assert ki.check(root) == []


def test_check_reports_a_stale_index(tmp_path: Path) -> None:
    root = three_shelf_repo(tmp_path)
    write(root, "llms.txt", EXPECTED_INDEX.replace("About zeta", "About zeta!"))
    assert ki.check(root) == ["llms.txt: stale; run scripts/knowledge_index.py"]


def test_check_reports_a_missing_index(tmp_path: Path) -> None:
    root = three_shelf_repo(tmp_path)
    assert ki.check(root) == ["llms.txt: missing; run scripts/knowledge_index.py"]


def test_check_reports_tree_violations_instead_of_index_state(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"landscape/a.md": entry_text("a", "landscape", related="[ghost]")})
    assert ki.check(root) == ["knowledge/landscape/a.md: related: no entry named 'ghost'"]


def test_write_index_writes_the_file_when_the_tree_is_valid(tmp_path: Path) -> None:
    root = three_shelf_repo(tmp_path)
    assert ki.write_index(root) == []
    assert (root / "llms.txt").read_text() == EXPECTED_INDEX


def test_write_index_leaves_the_file_alone_when_the_tree_is_broken(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"landscape/a.md": entry_text("a", "landscape", related="[ghost]")})
    write(root, "llms.txt", "old\n")
    assert ki.write_index(root) == ["knowledge/landscape/a.md: related: no entry named 'ghost'"]
    assert (root / "llms.txt").read_text() == "old\n"


def test_main_check_exit_codes_and_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = three_shelf_repo(tmp_path)
    assert ki.main(["--check", "--root", str(root)]) == 1
    assert capsys.readouterr().out == "llms.txt: missing; run scripts/knowledge_index.py\n"
    assert ki.main(["--root", str(root)]) == 0
    assert ki.main(["--check", "--root", str(root)]) == 0
    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_knowledge_index.py -q -k "render or check or write_index or main"`
Expected: FAIL with `AttributeError` for `render_index`.

- [ ] **Step 3: Implement rendering, check, and main**

Append to `scripts/knowledge_index.py`. Add `import argparse` and `import sys` beside the other imports.

```python
INDEX_HEADER = """# axial knowledge base

> Everything this repository has read about evaluating LLM applications, as one file
> per topic. Fetch a file by its path. Generated by scripts/knowledge_index.py; edit the
> files, not this index.
"""


def render_index(entries: list[Entry]) -> str:
    """The llms.txt text: one H2 per non-empty shelf, one bullet per entry."""
    parts = [INDEX_HEADER]
    for shelf in SHELVES:
        rows = [e for e in entries if e.shelf == shelf]
        if not rows:
            continue
        lines = [f"\n## {shelf}\n"]
        for e in rows:
            lines.append(
                f"- [{e.name}]({KNOWLEDGE_DIR}/{shelf}/{e.name}.md): "
                f"{e.description} ({e.updated.isoformat()})"
            )
        parts.append("\n".join(lines) + "\n")
    return "".join(parts)


def _render_or_violations(root: Path) -> tuple[str | None, list[str]]:
    try:
        return render_index(load_tree(root)), []
    except KnowledgeError as exc:
        return None, exc.violations


def check(root: Path) -> list[str]:
    """Every violation in the tree, or the one line saying the index is stale or missing."""
    rendered, problems = _render_or_violations(root)
    if rendered is None:
        return problems
    index = root / INDEX_NAME
    if not index.exists():
        return [f"{INDEX_NAME}: missing; run scripts/knowledge_index.py"]
    if index.read_text() != rendered:
        return [f"{INDEX_NAME}: stale; run scripts/knowledge_index.py"]
    return []


def write_index(root: Path) -> list[str]:
    """Rewrite llms.txt, or return the violations and leave it untouched."""
    rendered, problems = _render_or_violations(root)
    if rendered is None:
        return problems
    (root / INDEX_NAME).write_text(rendered)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="validate and compare; write nothing")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: the parent of scripts/)",
    )
    args = parser.parse_args(argv)
    problems = check(args.root) if args.check else write_index(args.root)
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_knowledge_index.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check scripts tests/test_knowledge_index.py
git add scripts/knowledge_index.py tests/test_knowledge_index.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
Render llms.txt from the knowledge tree and refuse to write it for a broken tree

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
```

---

### Task 4: The methodology shelf, `knowledge/README.md`, CI, and the real-tree test

This task creates the tree. From here on every content task ends with regenerating `llms.txt` and running the real-tree test.

**Files:**
- Create: `knowledge/README.md`, `knowledge/methodology/methodology-spine.md`, `knowledge/methodology/numbers-and-rules-of-thumb.md`, `knowledge/methodology/points-of-disagreement.md`, `knowledge/methodology/mechanizable-vs-human.md`, `llms.txt`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_knowledge_index.py`

**Interfaces:**
- Consumes: `check` from Task 3.
- Produces: the four methodology `name`s above, which procedures in Task 7 cite in `related`.

- [ ] **Step 1: Write the real-tree test**

Append to `tests/test_knowledge_index.py`:

```python
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_repository_knowledge_tree_is_valid_and_indexed() -> None:
    assert ki.check(REPO_ROOT) == []
```

Run: `uv run pytest tests/test_knowledge_index.py -q -k repository`
Expected: FAIL, `["knowledge: directory does not exist"]`.

- [ ] **Step 2: Write `knowledge/README.md`**

```markdown
# The knowledge base

Everything this repository has read about evaluating LLM applications, as one file per
topic, for agents and people to query. The query surface is [`llms.txt`](../llms.txt)
at the repository root: fetch it, pick a line, fetch that file.

## Shelves

- `methodology/`: what the practitioners whose work `axial` implements say to do, and
  where they disagree. Sources are their writing. Changes rarely.
- `procedures/`: a loop a team runs end to end, written as numbered steps, each with its
  trigger, its artifact and whether a human gates it. Cites landscape entries for the
  tools it uses and methodology entries for the numbers it quotes.
- `landscape/`: one tool per file. Claims decay, so every file carries the date its
  sources were read and the index shows it.

## Frontmatter

Every file starts with YAML frontmatter. Required: `name` (equals the filename stem,
kebab-case, unique across shelves), `description` (one line, at most 160 characters, no
trailing period; it is the index line, so say what the file answers), `shelf` (equals
the directory), `updated` (date of the last substantive change), `sources` (a list of
`{url, read}`). Optional: `related` (names of other files), and for landscape entries
`license` (SPDX identifier, or `proprietary`) and `extra` (the optional-dependency
group in `pyproject.toml` through which `axial` depends on the tool; absent means no
dependency).

Links between files are relative Markdown links so GitHub renders them. The check
resolves them.

Landscape entries, except `landscape-overview`, have the same seven headings in this
order: What it is, Core abstraction, Where it sits, Maturity, Biggest complaint,
Relation to axial, Corrections. A heading with nothing to say holds one line.
Corrections are dated notes, newest last; a corrected claim stays in the file.

## Commands

```bash
uv run python scripts/knowledge_index.py            # rewrite llms.txt
uv run python scripts/knowledge_index.py --check    # validate; CI runs this
```

The check reports every violation before exiting and never writes an index for a
broken tree. The rules above are the reader's version of
[`docs/superpowers/specs/2026-10-07-knowledge-base-layout-design.md`](../docs/superpowers/specs/2026-10-07-knowledge-base-layout-design.md);
when the schema changes, both change in the same commit.
```

- [ ] **Step 3: Split the methodology doc into four files**

The source is `docs/research/2026-09-17-methodology.md`. Its H2 sections sit at these lines (verify with `grep -n "^## " docs/research/2026-09-17-methodology.md` before cutting):

| Target file | Source lines | Title line to write |
|---|---|---|
| `methodology-spine.md` | 7 to 61 | `# The methodology spine` |
| `numbers-and-rules-of-thumb.md` | 62 to 111 | `# Concrete numbers and rules of thumb` |
| `points-of-disagreement.md` | 112 to 142 | `# Points of disagreement` |
| `mechanizable-vs-human.md` | 143 to 211 | `# Mechanizable in Python vs irreducibly human` |

For each file: write the frontmatter below, then the title line, then a blank line, then the source lines with the original `## N. Title` heading line removed (the title line replaces it) and every `###` heading promoted to `##`. Do not reword the body. Prefix each with the frontmatter from the table below; `read` is `2026-09-17` for every source.

Sources: the source doc's `## 5. Sources` section (lines 212 to end) lists every URL by author. Each file lists, in its frontmatter, every source whose author or title the file's body names. When a section names nobody, list Husain's FAQ, `https://hamel.dev/blog/posts/evals-faq/`, which the source doc calls its primary source. When unsure whether a section draws on a source, include it.

Frontmatter for each (fill `sources` by the rule above):

```yaml
---
name: methodology-spine
description: The thirteen-step error-analysis-first loop from instrumentation to production monitoring, with each step's artifact
shelf: methodology
updated: 2026-09-17
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
---
```

```yaml
---
name: numbers-and-rules-of-thumb
description: Effort, cadence, sample sizes and agreement thresholds the practitioners quote, and who quotes which
shelf: methodology
updated: 2026-09-17
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
---
```

```yaml
---
name: points-of-disagreement
description: Where the practitioner sources disagree on method, numbers and judge design
shelf: methodology
updated: 2026-09-17
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
---
```

```yaml
---
name: mechanizable-vs-human
description: Which eval steps a tool should own outright, which must stay human, and the grey zone between
shelf: methodology
updated: 2026-09-17
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
---
```

Add `related: [numbers-and-rules-of-thumb, mechanizable-vs-human]` to the spine's frontmatter and `related: [methodology-spine]` to the other three.

The source doc's `## 5. Sources` block and its two-line preamble (lines 1 to 5) are not copied anywhere; the frontmatter replaces them.

- [ ] **Step 4: Generate the index and run the real-tree test**

```bash
uv run python scripts/knowledge_index.py
uv run pytest tests/test_knowledge_index.py -q
```

Expected: the script prints nothing and exits 0, `llms.txt` exists with a `## methodology` section of four lines, and all tests pass. If the script prints violations, fix the frontmatter it names and rerun.

- [ ] **Step 5: Add the CI job**

In `.github/workflows/ci.yml`, after the `compat` job, add:

```yaml
  knowledge:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.14"
      - run: uv sync
      - run: uv run python scripts/knowledge_index.py --check
```

- [ ] **Step 6: Commit**

```bash
uv run ruff check .
git add knowledge llms.txt .github/workflows/ci.yml tests/test_knowledge_index.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
Start the knowledge base: methodology shelf, generated llms.txt, CI check

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
```

`docs/research/2026-09-17-methodology.md` is deleted in Task 8, after the landscape doc is migrated too, so the two deletions and the README change land together.

---

### Task 5: Split the tooling landscape into one file per tool

**Files:**
- Create: `knowledge/landscape/landscape-overview.md` and the nineteen tool files listed below.
- Modify: `llms.txt` (regenerated)

**Interfaces:**
- Consumes: `check` from Task 3.
- Produces: the landscape `name`s below, cited by Task 6 and Task 7.

- [ ] **Step 1: Write `landscape-overview.md`**

Frontmatter:

```yaml
---
name: landscape-overview
description: Two families of eval tools, benchmarking and application evals, and why none closes the error-analysis-first gap
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed.html
    read: 2026-09-17
  - url: https://arxiv.org/abs/2404.12272
    read: 2026-09-17
  - url: https://www.braintrust.dev/articles/best-promptfoo-alternatives-2026
    read: 2026-09-17
related: [methodology-spine, evalgen, evals-skills]
---
```

Body: title `# The landscape of eval tooling`, then from `docs/research/2026-09-17-tooling-landscape.md` copy, unchanged: the `## Two different problems, one blurred vocabulary` section; the sentence under the comparison table beginning "Notes on things that were hard to pin"; and the whole `## Synthesis: where are the genuine gaps?` section. Where the synthesis names a tool, link its first mention to the tool's file, for example `[Inspect AI](inspect-ai.md)`. Add after the synthesis a `## Index of entries` section with one bullet per tool file created in this task, as `- [name](name.md)`. The comparison table itself is not copied; the per-tool files replace it.

- [ ] **Step 2: Write one file per tool**

For each row of the comparison table in `docs/research/2026-09-17-tooling-landscape.md`, except the Evidently row (rewritten in Task 6) and the standalone autoevals row (folded into `braintrust-autoevals`), create `knowledge/landscape/<name>.md` with the frontmatter from the table below and a body built by this mapping. Copy cell text unchanged; do not summarise it.

| Heading | Source |
|---|---|
| `# <Tool as the table names it>` | first column |
| `## What it is` | "What it is" column |
| `## Core abstraction` | "Core abstraction you write" column |
| `## Where it sits` | "Benchmarking vs. application evals" column, then a second paragraph from "Local-first vs. account" |
| `## Maturity` | "Maturity / last release" column, prefixed `As of 2026-09-17: ` |
| `## Biggest complaint` | "Biggest practitioner complaint" column |
| `## Relation to axial` | the bullet in the synthesis that names this tool, copied; then the line `No dependency.` |
| `## Corrections` | `None.` unless the table below says otherwise |

Frontmatter template, with the per-tool values from the table after it. `read` is `2026-09-17` for every source unless the table says otherwise. `related: [landscape-overview]` on every file.

```yaml
---
name: <name>
description: <description>
shelf: landscape
updated: <updated>
sources:
  - url: <each source URL>
    read: 2026-09-17
license: <license>
related: [landscape-overview]
---
```

| name | description | license | updated | sources (from the doc's Sources list) |
|---|---|---|---|---|
| `promptfoo` | CLI and Node library for prompt and model testing and red-teaming, driven by a YAML config of prompts, providers, tests and asserts | `MIT` | 2026-09-17 | promptfoo GitHub releases; promptfoo release notes |
| `inspect-ai` | UK AISI's open harness for model and agent evals, with datasets, solvers, scorers, sandboxed agents and a local log viewer | `MIT` | 2026-10-07 | inspect.aisi.org.uk; inspect_ai repo; inspect_evals repo |
| `deepeval` | Pytest-style LLM eval library with 60+ built-in metrics and a paid Confident AI dashboard above it | `Apache-2.0` | 2026-09-17 | DeepEval PyPI; Confident AI review 2026 |
| `ragas` | RAG-specific evaluation and synthetic test-set generation over question, answer, contexts and ground truth | `Apache-2.0` | 2026-09-17 | Ragas docs; vibrantlabsai/ragas |
| `openai-evals` | OpenAI's slowing open-source eval registry and the hosted Evals API its docs now steer users to | `MIT` | 2026-09-17 | openai/evals; Getting Started with OpenAI Evals |
| `braintrust-autoevals` | Braintrust's closed eval platform and autoevals, its open scorer library for heuristic, embedding and LLM-judge scores | `proprietary` | 2026-09-17 | braintrustdata/autoevals; autoevals PyPI |
| `langsmith` | LangChain's closed eval and tracing platform, with open evaluator classes in the client libraries | `proprietary` | 2026-09-17 | LangSmith evaluation docs; LangSmith alternatives 2026 |
| `wandb-weave` | Weights & Biases tracing and evaluation SDK, usable only with a W&B account for storage and the UI | `Apache-2.0` | 2026-09-17 | wandb/weave; weave PyPI |
| `arize-phoenix` | Arize's OpenTelemetry-based observability and eval library, source-available under Elastic License 2.0 | `Elastic-2.0` | 2026-09-17 | Arize-ai/phoenix; Phoenix license docs; Phoenix relicensing issue #9890 |
| `helm` | Stanford CRFM's holistic foundation-model benchmark framework, in maintenance mode since June 2026 | `Apache-2.0` | 2026-09-17 | stanford-crfm/helm |
| `lm-evaluation-harness` | EleutherAI's standard harness for reproducible few-shot benchmark evaluation of foundation models | `MIT` | 2026-09-17 | EleutherAI/lm-evaluation-harness; lm-eval PyPI |
| `anthropic-cookbook` | Anthropic's eval notebooks and guidance, rubrics and Claude as judge; recipes to copy, nothing to install | `MIT` | 2026-09-17 | anthropics/claude-cookbooks building_evals.ipynb |
| `pytest-evals` | A thin pytest plugin for running and aggregating LLM eval cases as ordinary tests in existing CI | `MIT` | 2026-09-17 | AlmogBaku/pytest-evals |
| `dspy` | Stanford's framework that treats an eval metric as the optimisation objective for prompts and few-shots | `MIT` | 2026-09-17 | DSPy metrics and evaluation docs; stanfordnlp/dspy metrics.md |
| `langfuse` | Open-source LLM engineering platform for tracing, datasets, evals and annotation queues; MIT since June 2025 | `MIT` | 2026-09-17 | Langfuse Doubling Down on Open Source; langfuse/langfuse |
| `opik` | Comet's open-source observability and evaluation platform with pytest integration and a prompt optimizer | `Apache-2.0` | 2026-09-17 | comet-ml/opik; Opik PyPI |
| `pydantic-evals` | Pydantic AI's typed Case, Dataset, Evaluator and LLMJudge, with no judge-to-human agreement measurement | `MIT` | 2026-10-07 | the five URLs under the Pydantic Evals addendum, read 2026-09-18 |
| `evals-skills` | The AI Evals course's agent skills, error-discovery and validate-evaluator, which count Pass as the positive class | `Apache-2.0` | 2026-10-07 | github.com/ai-evals-course/evals-skills read 2026-10-07; the ai-digest topic page read 2026-09-17 |
| `evalgen` | Shankar et al.'s research prototype that picks judge prompts by agreement with human grades and names criteria drift | omit the key | 2026-09-17 | arxiv.org/abs/2404.12272 |

Special cases:

- `braintrust-autoevals`: the `## What it is` paragraph ends with the sentence `autoevals itself is MIT; the licence above is the platform's.`
- `inspect-ai`: `## Corrections` holds the dated note from the table's synthesis bullet: `2026-09-30: the first version of this survey filed Inspect with lm-eval-harness and HELM as benchmark-only. Corrected after reading its scorer and scoring-workflow docs on 2026-10-07; see Relation to axial.` Its `## Relation to axial` copies the Inspect bullet from the synthesis and the Inspect paragraph from the README's Prior art section, then `No dependency.`
- `pydantic-evals`: the body is the addendum's four paragraphs mapped onto the seven headings. `## What it is` and `## Core abstraction` from the addendum's "What it is" and "Core abstractions" paragraphs; `## Where it sits` is `Application evals. Fully local; Logfire is optional and pydantic-ai is not required.`; `## Maturity` is `As of 2026-09-18: MIT, about 20k stars on the parent repo, directional.`; `## Biggest complaint` is the "What it does not have" list; `## Relation to axial` is the "Verdict" paragraph plus `No dependency.`; `## Corrections` is `None.`
- `evals-skills`: `## What it is` is the README's Prior art paragraph about `evals-skills`; `## Core abstraction` is `Agent skills (SKILL.md files) installed with npx skills add.`; `## Where it sits` is `Application evals, as instructions an agent follows rather than a library.`; `## Maturity` is `As of 2026-10-07: history in the repo starts 2026-08-15 with commits by Hamel Husain.`; `## Biggest complaint` is `Not applicable.`; `## Relation to axial` is `axial is the versioned artifact store and CI gate around the same ideas. The skills count Pass as the positive class, so their TPR is axial's TNR. No dependency.`; `## Corrections` holds the two dated notes from the source doc's "Other sources" section, the 2026-09-17 digest attribution and the 2026-10-07 correction, each prefixed with its date.
- `evalgen`: `## What it is` is `A research prototype (Shankar, Zamfirescu-Pereira, Hartmann, Parameswaran, Arawjo, UIST 2024) that generates candidate judge prompts, uses human grades on a subset to select among them, and names criteria drift: graders' criteria change as they read outputs.`; `## Core abstraction` is `A candidate set of judge criteria graded against human labels.`; `## Where it sits` is `Application evals. A paper and prototype, not a package.`; `## Maturity` is `As of 2026-09-17: not maintained as software.`; `## Biggest complaint` is `Not applicable.`; `## Relation to axial` is the README's Prior art paragraph about EvalGen plus `No dependency.`; `## Corrections` is `None.`
- `anthropic-cookbook`: `## Relation to axial` is `axial implements the rubric-plus-judge guidance the cookbook gives as notebooks, and adds the alignment measurement the notebooks do not. No dependency.`
- Every other tool: `## Relation to axial` is the synthesis bullet that names it (the bullets group tools; copy the bullet the tool appears in) followed by `No dependency.`

- [ ] **Step 3: Generate the index and run the checks**

```bash
uv run python scripts/knowledge_index.py
uv run pytest tests/test_knowledge_index.py -q
```

Expected: no violations, twenty lines under `## landscape`, tests pass. A violation naming a description over 160 characters means a description from the table was altered; restore it.

- [ ] **Step 4: Commit**

```bash
git add knowledge llms.txt
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
Split the tooling landscape into one dated entry per tool

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
```

---

### Task 6: New landscape entries: Warp Factories, Great Expectations, Evidently

**Files:**
- Create: `knowledge/landscape/warp-factories.md`, `knowledge/landscape/great-expectations.md`, `knowledge/landscape/evidently.md`
- Modify: `llms.txt` (regenerated)

**Interfaces:**
- Consumes: `check`; the names `landscape-overview`, `methodology-spine`.
- Produces: the three names above, cited by Task 7.

- [ ] **Step 1: Write `warp-factories.md`**

```markdown
---
name: warp-factories
description: Warp's agent factory as files, with foreman, triage, spec, implement and review agents, automations, runners and self-grading scorers
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://github.com/warpdotdev/warp-factory-examples
    read: 2026-10-07
  - url: https://docs.warp.dev/factories/
    read: 2026-10-07
license: proprietary
related: [landscape-overview, factory-lifecycle, methodology-spine]
---
# Warp Factories

## What it is

Warp's hosted software factory: a team of agents that takes a request from Slack,
Linear or GitHub to a review-ready pull request, defined entirely as files in a Git
repository. The product is proprietary; the example definitions are MIT and are the
source for everything below. Warp's own account of running it internally reports the
cost per PR falling from 80 to 30 dollars; that figure is from a secondary write-up of
Ben Holmes's post and was not verified against Warp's own text.

## Core abstraction

A definition root containing `factory.yaml` (repositories, integrations, agent
defaults), `agents/<name>/agent.md` (a prompt with frontmatter: `agentType` of
FOREMAN, TRIAGE, SPEC, IMPLEMENT, REVIEW or CUSTOM, an optional per-agent `model` or
`harness`, an optional `runner`), `automations/<name>/automation.md` (event triggers
with AND-combined filters, plus a prompt), `runners/<name>.yaml` (compute shapes and
images), `scorers/<name>/scorer.md` and skills at factory or agent scope. A resource's
name is its path. Exactly one agent is the foreman. A validator script checks a tree
before it is registered.

## Where it sits

An agent-platform scorer and orchestration product, not an eval library. The scorers
are the part that concerns evals: each declares the agents it samples, a label set with
a score per label, a `passingScore`, a `samplingRate` in percent, a judge `model`, and
an optional `selfImprovement` flag that feeds verdicts back into the agent it grades.

Account-required and hosted; execution can run on a self-hosted worker.

## Maturity

As of 2026-10-07: schema `v1alpha1`, early access, eight published example trees.

## Biggest complaint

Not applicable; too new for a practitioner consensus. The design observation is in the
next section.

## Relation to axial

The scorers are LLM judges and nothing in the definition validates one against human
labels before `selfImprovement` is turned on. The full-lifecycle example samples a
quarter of implementer runs with a three-label task-compliance scorer, self-improvement
on, and a quarter of reviewer runs with a review-quality scorer, both on `model:
claude-4-5-haiku`, a name without a version date that axial would refuse as a floating
alias under its no-floating-aliases invariant. A self-improving agent steered by an
unmeasured judge drifts toward whatever the judge rewards. axial is the step that
measures the judge first. The two label sets are three-valued where axial's judge is
binary, so a port needs a human decision about the middle label; no exporter exists.
No dependency.

The procedure the examples define is written up in
[factory-lifecycle](../procedures/factory-lifecycle.md).

## Corrections

None.
```

- [ ] **Step 2: Write `great-expectations.md`**

```markdown
---
name: great-expectations
description: Data-quality contracts for pipelines, with expectation suites, checkpoints that stop a run, and Data Docs as the record
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://greatexpectations.io/
    read: 2026-10-07
license: Apache-2.0
related: [landscape-overview, data-contracts-at-ingest]
---
# Great Expectations

## What it is

GX Core, an open-source Python framework for asserting what data should look like and
validating it where a pipeline runs, plus GX Cloud, a hosted layer over it. Its home
page calls it the most popular data quality framework in the world, with a community
of over 11,000 practitioners.

## Core abstraction

An Expectation is one testable assertion about data. An Expectation Suite groups them.
Validation runs a suite against a batch of data. A Checkpoint is the orchestration
point that runs validations and acts on the result. Data Docs are generated,
human-readable pages of what was checked and what passed.

## Where it sits

Data quality for tabular pipelines, not LLM evaluation. Checkpoints run before a
pipeline step to reject bad input and after it to verify the output, stop the run when
a threshold is missed, and notify. Integrations named on the home page: Airflow,
Databricks, BigQuery, Snowflake, PostgreSQL, S3, Azure Blob Storage, Jupyter.

GX Core is local-first and needs no account; GX Cloud is the hosted option.

## Maturity

As of 2026-10-07: GX Core is Apache-2.0 and actively released; GX Cloud is a commercial
product.

## Biggest complaint

Not applicable; not surveyed for practitioner complaints, since it is outside the LLM
eval family.

## Relation to axial

The same loop at the data layer. An expectation suite is the contract, a checkpoint
is the gate, Data Docs are the record. In axial the contract is the labelled split, the
gate is `axial gate`, and `alignment.md` is the record. What axial could take: an
expectation-style check at ingest for empty outputs, duplicate trace ids and timestamp
ranges, so a bad export fails before it reaches a review batch. See
[data-contracts-at-ingest](../procedures/data-contracts-at-ingest.md). No dependency.

## Corrections

None.
```

- [ ] **Step 3: Write `evidently.md`**

```markdown
---
name: evidently
description: Open-source ML and LLM evaluation and monitoring; Reports and Tests, descriptors, LLM judges, tracing, scheduled evals
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://docs.evidentlyai.com/llms.txt
    read: 2026-10-07
  - url: https://www.evidentlyai.com/
    read: 2026-10-07
  - url: https://github.com/evidentlyai/evidently
    read: 2026-09-17
  - url: https://www.evidentlyai.com/llm-guide/llm-evaluation
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview, evals-in-ci, production-monitoring]
---
# Evidently

## What it is

An open-source Python library for evaluating and monitoring both classic ML systems
(data drift, data quality, predictive performance) and LLM applications (chatbots, RAG,
agents), with over 100 built-in metrics and a declarative testing API, plus a platform,
self-hosted or Evidently Cloud, for storing runs, datasets, traces and dashboards. Its
documentation is published as an `llms.txt` index of one page per topic, which this
knowledge base copies.

## Core abstraction

A Report runs metrics over a dataframe and summarises them. Tests are the same metrics
with pass or fail conditions attached. Descriptors are row-level evaluations of text,
including LLM-as-judge descriptors configured by a prompt. Datasets hold test cases, and
a synthetic data API generates edge-case and adversarial inputs. Tracely captures live
LLM inputs and outputs for later evaluation.

## Where it sits

Mixed: application evals bolted onto a larger ML-monitoring product, and the only tool
in this shelf with first-class production monitoring. The documented workflows are
evals in GitHub Actions, scheduled evals on the platform, batch monitoring jobs over
traced traffic, dashboards and alerts.

The library is fully local; the platform is self-hosted or cloud.

## Maturity

As of 2026-10-07: Apache-2.0, 7,500 or more GitHub stars and 40 million downloads by
its own count, directional; active, with a broad user base carried over from its
ML-monitoring roots.

## Biggest complaint

LLM eval feels like a secondary feature grafted onto a data-drift-monitoring tool, with
less depth on LLM-specific failure modes, such as hallucination nuance and agent
trajectories, than dedicated LLM-eval libraries.

## Relation to axial

Evidently ships a catalogue of pre-named metrics and asks you to write a judge prompt;
it does not derive the judge from your failures or measure its agreement with your
humans. axial produces the judge Evidently would run. What axial takes from it: the
index pattern for this knowledge base, Tests as the pass-or-fail form of a Report, and
the two procedures [evals-in-ci](../procedures/evals-in-ci.md) and
[production-monitoring](../procedures/production-monitoring.md). A writer that emits
axial verdicts as an Evidently descriptor column, so their dashboards can show a
validated judge, is a possible optional extra, not built. No dependency.

## Corrections

None.
```

- [ ] **Step 4: Regenerate, check, commit**

The three `related` lists name procedure files that do not exist until Task 7, and the bodies link to them. The check fails on those links and names until Task 7 lands. So run only the script's parse on these three files now:

```bash
uv run python -c "
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('ki', 'scripts/knowledge_index.py'); ki = importlib.util.module_from_spec(spec); spec.loader.exec_module(ki)
root = pathlib.Path('.')
for n in ('warp-factories', 'great-expectations', 'evidently'):
    print(ki.parse_entry(root / 'knowledge/landscape' / f'{n}.md', root).name)
"
```

Expected: the three names printed, no `KnowledgeError`. Do not regenerate `llms.txt` yet; commit the three files alone. The real-tree test is expected to fail between this commit and the next.

```bash
git add knowledge/landscape/warp-factories.md knowledge/landscape/great-expectations.md knowledge/landscape/evidently.md
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
Add Warp Factories and Great Expectations to the landscape; rewrite Evidently

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
```

---

### Task 7: The four procedures

**Files:**
- Create: `knowledge/procedures/factory-lifecycle.md`, `knowledge/procedures/evals-in-ci.md`, `knowledge/procedures/production-monitoring.md`, `knowledge/procedures/data-contracts-at-ingest.md`
- Modify: `llms.txt` (regenerated)

**Interfaces:**
- Consumes: `check`; the names from Tasks 4 to 6.
- Produces: the four procedure names, after which the whole tree validates.

Every procedure is numbered steps. Each step has three labelled lines: **Trigger**, **Artifact**, **Gate** (`none`, or who decides).

- [ ] **Step 1: Write `factory-lifecycle.md`**

```markdown
---
name: factory-lifecycle
description: Issue to merged PR with agents doing every stage and humans at two gates, as the Warp factory examples define it
shelf: procedures
updated: 2026-10-07
sources:
  - url: https://github.com/warpdotdev/warp-factory-examples/tree/main/examples/02-sdlc-issue-to-pr
    read: 2026-10-07
  - url: https://github.com/warpdotdev/warp-factory-examples/tree/main/examples/05-ui-verification
    read: 2026-10-07
  - url: https://github.com/warpdotdev/warp-factory-examples/tree/main/examples/06-common-automations
    read: 2026-10-07
related: [warp-factories, production-monitoring, evals-in-ci]
---
# The factory lifecycle

A standing engineering loop in which agents do every stage and a person decides only
what gets built and what ships. Taken from the full-lifecycle, UI-verification and
common-automations examples of [Warp Factories](../landscape/warp-factories.md). Every
stage starts from an event and ends by emitting an artifact that is itself the next
trigger; nothing runs on a schedule except standing audits.

## Steps

1. **Intake.**
   Trigger: a Slack mention or DM, a Linear agent session, or a GitHub mention or
   assignment on an issue carrying the factory's routing label. Filters combine with
   AND, so the label is the volume control.
   Artifact: a work item the foreman owns. The foreman first checks the event is
   addressed to this factory and stops silently when it is not.
   Gate: none.
2. **Triage.**
   Trigger: the foreman dispatches it whenever cause or scope is unknown, the work needs
   a tracked issue, or it may duplicate existing work. When in doubt, triage.
   Artifact: a Linear issue that is the durable record: the request in the requester's
   words, findings, code locations, reproduction steps for bugs. Plus a complexity
   estimate and the ambiguities a spec would need to resolve. Runs on a cheap model.
   Gate: none.
3. **Spec.**
   Trigger: triage reported the work as ambiguous or large.
   Artifact: the smallest Markdown spec that resolves the ambiguity, with desired
   behaviour, non-goals and validation criteria, committed in a draft PR on a branch
   named for the work item.
   Gate: a human approves the spec. The agent never approves on a human's behalf.
4. **Implement.**
   Trigger: the foreman dispatches with the work item and, when one exists, the spec PR.
   Artifact: the change, tests for the changed behaviour, and test and linter output
   pasted into the PR description, in the spec's own PR and branch. Runs on a larger
   runner with the toolchain image. Scope growth is reported, not absorbed.
   Gate: none.
5. **Review.**
   Trigger: the implementation PR exists.
   Artifact: a findings report to the foreman, each finding with file and line,
   severity, problem, impact and correction, classified as unambiguous or a judgment
   call. The reviewer runs on a different model from the implementer so the two do not
   share blind spots, reviews against the issue and spec rather than style, and checks
   that the tests exercise the changed behaviour. The foreman relays unambiguous
   findings back to the implementer until the PR is clean.
   Gate: judgment calls wait for a human.
6. **Verify.**
   Trigger: a PR labelled for a UI check, or an issue labelled for reproduction.
   Artifact: a comment with screenshots or a recording showing the behaviour, made by
   checking out the branch, running the app and driving it in a browser. The verifier
   never pushes fixes.
   Gate: none.
7. **Ship.**
   Trigger: review is clean and verification posted.
   Artifact: the merged PR.
   Gate: a human merges. The foreman never merges.
8. **Close the loop.**
   Trigger: a factory-labelled PR is merged or closed.
   Artifact: merged, the linked issues move to done; closed without merging, a report
   on the issue instead.
   Gate: none.
9. **Monitor.**
   Trigger: a CI workflow fails on the default branch.
   Artifact: a fix PR when the cause is small and clear, otherwise an issue with the
   failing step, the error and the commit range. That issue is step one's input.
   Gate: none.
10. **Grade the agents.**
    Trigger: a sampled fraction of implementer and reviewer runs.
    Artifact: a scorer verdict per run: did the PR deliver exactly what the issue asked,
    judged from the issue text; did each review finding carry a location, impact and
    correction. The implementer scorer feeds its verdicts back into the agent.
    Gate: none, which is the problem the next section names.

## What to keep and what to add

Keep the explicit "you do not do X" clause on every agent, so handoffs are the only way
work moves, and the routing label, so a person can inject work anywhere by applying it.

Add a measurement of the scorers before self-improvement is turned on. The scorers are
LLM judges with no recorded agreement against human labels; see
[warp-factories](../landscape/warp-factories.md). Validating one is the
[methodology spine](../methodology/methodology-spine.md) from step seven onward, and
feeding production back into it is [production-monitoring](production-monitoring.md).
```

- [ ] **Step 2: Write `evals-in-ci.md`**

```markdown
---
name: evals-in-ci
description: Gate a pull request on a promoted judge and keep the gate honest with deterministic checks first, intervals not points, and a saturation watch
shelf: procedures
updated: 2026-10-07
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
  - url: https://anthropic.com/engineering/demystifying-evals-for-ai-agents
    read: 2026-09-17
  - url: https://docs.evidentlyai.com/examples/GitHub_actions.md
    read: 2026-10-07
related: [methodology-spine, numbers-and-rules-of-thumb, evidently, production-monitoring]
---
# Evals in CI

Step eleven of the [methodology spine](../methodology/methodology-spine.md) as a
procedure. The contract is a frozen regression set and a threshold; the check runs on
every change; the gate blocks the merge. The numbers quoted are collected in
[numbers-and-rules-of-thumb](../methodology/numbers-and-rules-of-thumb.md).

## Steps

1. **Freeze the regression set.**
   Trigger: a judge is promoted, or a bug is fixed and its trace becomes a case.
   Artifact: a small purpose-built dataset, a hundred or more examples, covering core
   features, fixed bugs and known edge cases. In axial this is the labelled split and
   the promoted judge directory under `.axial/judges/`, committed with the code.
   Gate: a human decides what goes in. The test split is write-once.
2. **Prefer deterministic assertions.**
   Trigger: writing a check.
   Artifact: a code assertion wherever one can express the failure, because CI runs
   constantly and judge calls cost money. A judge only for failure modes code cannot
   express.
   Gate: none.
3. **Run on every change and every model upgrade.**
   Trigger: a pull request, and any change to the model behind the system.
   Artifact: a verdict per case. For non-deterministic systems, run each case several
   times and report pass@k when one working answer is enough or pass^k when every run
   must succeed.
   Gate: none.
4. **Gate on an interval, not a point.**
   Trigger: the verdicts exist.
   Artifact: a pass or fail exit code. Two runs of the same system differ by a few
   points routinely, so a gate set inside that noise fails builds at random until
   someone disables it. axial's `gate` compares a paired bootstrap interval against a
   baseline and, with the batch mode the corrected-rates spec adds, a corrected failure
   rate against a maximum or against a baseline batch.
   Gate: the exit code blocks the merge.
5. **Watch for saturation.**
   Trigger: a suite stops failing anything.
   Artifact: a warning that the eval has stopped carrying signal, which axial's gate
   prints. Retire or replace the suite.
   Gate: a human decides what replaces it.

## The Evidently form

Evidently documents the same loop as a GitHub Actions job: load the dataset, run Tests
with pass or fail conditions, upload the result to the platform, fail the job on a
failed test. See [evidently](../landscape/evidently.md). The difference is upstream:
Evidently's judge is a prompt you wrote; axial's is one whose agreement with your
labels was measured before it was allowed into this job.
```

- [ ] **Step 3: Write `production-monitoring.md`**

```markdown
---
name: production-monitoring
description: Sample live traces, measure them with a promoted judge, alert on the interval's lower bound, and feed flagged traces back into review
shelf: procedures
updated: 2026-10-07
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
  - url: https://docs.evidentlyai.com/docs/platform/monitoring_scheduled_evals.md
    read: 2026-10-07
  - url: https://docs.evidentlyai.com/docs/platform/monitoring_local_batch.md
    read: 2026-10-07
  - url: https://github.com/warpdotdev/warp-factory-examples/tree/main/examples/06-common-automations
    read: 2026-10-07
related: [methodology-spine, evals-in-ci, evidently, warp-factories, factory-lifecycle]
---
# Production monitoring

Steps twelve and thirteen of the [methodology spine](../methodology/methodology-spine.md)
as a procedure. Production has no reference outputs, so it leans on a reference-free
judge run asynchronously over sampled live traces, and the loop closes when what the
judge flags is read by a person and promoted into the CI set.

## Steps

1. **Collect traces.**
   Trigger: live traffic.
   Artifact: a trace store with the complete record of each request: messages, tool
   calls, retrievals. Evidently's Tracely and any OpenTelemetry pipeline produce this;
   axial ingests JSONL or CSV through an adapter.
   Gate: none.
2. **Sample a batch.**
   Trigger: a schedule, daily or weekly, matched to traffic.
   Artifact: a batch mixing a random slice with targeted slices, because every targeted
   method has a bias and the random slice is where unknown failures appear.
   Gate: none.
3. **Measure with a promoted judge.**
   Trigger: the batch exists.
   Artifact: a corrected failure rate with a confidence interval. A judge with known
   true-positive and true-negative rates on its test split gives a raw flag rate that is
   neither the failure rate nor an interval; the correction and interval are what the
   corrected-rates spec adds to axial as `measure`. Evidently's scheduled evals and
   batch monitoring jobs run the equivalent for its own descriptors.
   Gate: none.
4. **Alert on the lower bound.**
   Trigger: the interval's lower bound crosses the threshold.
   Artifact: an alert, and from it a ticket. The common-automations example of
   [Warp Factories](../landscape/warp-factories.md) shows the ticket form: an issue with
   the failing step, the error and the commit range, which re-enters intake. A soft
   warning is not monitoring.
   Gate: none.
5. **Promote flagged traces into review.**
   Trigger: the measurement exists.
   Artifact: a review batch drawn from the measured batch, mixing flagged and unflagged
   traces, that a person annotates. New failure modes found here go through open
   coding and into the taxonomy; fixed ones become CI cases.
   Gate: a human reads them. This is the irreducibly human step.
6. **Re-check the judge on what it measured.**
   Trigger: the labels from step five exist.
   Artifact: the judge's true-positive and true-negative rates on this batch beside the
   rates from its test split. The correction in step three assumed they still hold;
   production drifts, so this is the check on the checker.
   Gate: a human decides whether the judge needs a new version, which starts the
   alignment loop again.

## Drift

A batch that looks unlike the data the judge was aligned on is a reason to distrust
step three before any alert fires. Evidently detects it with embedding and
distribution drift over text; axial has a character-level perplexity model trained on
the ingested corpus that can score a new batch for the same purpose, as a warning and
never a verdict.
```

- [ ] **Step 4: Write `data-contracts-at-ingest.md`**

```markdown
---
name: data-contracts-at-ingest
description: Expectation, checkpoint and data docs from Great Expectations mapped onto contract, gate and alignment report, with checks for axial ingest
shelf: procedures
updated: 2026-10-07
sources:
  - url: https://greatexpectations.io/
    read: 2026-10-07
related: [great-expectations, methodology-spine]
---
# Data contracts at ingest

The data-quality loop as [Great Expectations](../landscape/great-expectations.md)
defines it, and what it says about the traces an eval pipeline consumes. The point is
that a contract is checked where data enters, and a failed check stops the run rather
than warning.

## The mapping

| Great Expectations | What it is | In axial |
|---|---|---|
| Expectation | one testable assertion about data | one labelled failure mode |
| Expectation Suite | the assertions that define acceptable data | the labelled split for a mode |
| Validation | running a suite against a batch | `judge` on a split, `measure` on a batch |
| Checkpoint | the orchestration point that runs validations and acts | `gate`, and its exit code |
| Data Docs | generated human-readable record of each run | `alignment.md` and `alignment.json` |

## Steps

1. **Write the expectations before the data arrives.**
   Trigger: a new trace source is wired up.
   Artifact: the contract for a trace: required fields present, input and output
   non-empty, trace ids unique, timestamps inside the export's range, metadata keys
   known. The [methodology spine](../methodology/methodology-spine.md) calls the trace
   the unit everything else depends on.
   Gate: a human writes the contract.
2. **Validate on entry.**
   Trigger: `axial ingest` reads a file.
   Artifact: a pass, or a list of every violation with its row. Today ingest fails on
   the first missing required field; the checks above would run before any trace is
   written and report all of them.
   Gate: a failed contract stops the ingest. Nothing is written.
3. **Validate on exit.**
   Trigger: a review batch, a split or a measurement is written.
   Artifact: the same contract applied to the artifact: a split with every trace id
   resolvable, a batch with no duplicates, a measurement whose traces all parsed.
   Gate: a failed check stops the command.
4. **Keep the record human-readable.**
   Trigger: any validation.
   Artifact: a Markdown report of what was checked and what failed, committed beside
   the data, as Data Docs are generated beside a pipeline.
   Gate: none.
5. **Alert and ticket.**
   Trigger: a failed validation in a scheduled job.
   Artifact: a notification that becomes a work item, as in
   [production-monitoring](production-monitoring.md).
   Gate: none.
```

- [ ] **Step 5: Regenerate the index and run every check**

```bash
uv run python scripts/knowledge_index.py
uv run python scripts/knowledge_index.py --check
uv run pytest -q
uv run ruff check .
```

Expected: no violations, four lines under `## procedures`, the whole suite passes. A `link: ... does not exist` violation means a relative link in a procedure or a Task 6 entry points at a file that is not there; fix the link, not the file.

- [ ] **Step 6: Commit**

```bash
git add knowledge llms.txt
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
Write the four procedures: factory lifecycle, evals in CI, production monitoring, data contracts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
```

---

### Task 8: Retire `docs/research/` and point the README at the knowledge base

**Files:**
- Delete: `docs/research/2026-09-17-methodology.md`, `docs/research/2026-09-17-tooling-landscape.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the complete tree from Tasks 4 to 7.
- Produces: nothing further.

- [ ] **Step 1: Confirm nothing was lost in migration**

Every URL in the two source docs must appear in some knowledge file's frontmatter or body, since the sources were the point of the research. Run:

```bash
grep -ohE 'https?://[^ )>\]]+' docs/research/*.md | sort -u > /tmp/old-urls.txt
grep -rohE 'https?://[^ )>\]]+' knowledge | sort -u > /tmp/new-urls.txt
comm -23 /tmp/old-urls.txt /tmp/new-urls.txt
```

Expected: no output. Each URL printed is one that was dropped; add it to the frontmatter of the file whose body cites that author or tool, then regenerate the index. The URL `https://andresrubio.github.io/ai-digest/topics/evals-and-benchmarks.html` belongs in `evals-skills.md`.

- [ ] **Step 2: Delete the research docs**

```bash
git rm docs/research/2026-09-17-methodology.md docs/research/2026-09-17-tooling-landscape.md
```

- [ ] **Step 3: Update the README**

In `README.md`, the Credit section ends with the sentence beginning `See [`docs/research/`](docs/research/) for the sources behind every default.` Replace that sentence with `See the knowledge base below for the sources behind every default.` and add this section immediately after the Credit section and before Licence:

```markdown
## Knowledge base

Everything this repository has read about evaluating LLM applications lives under
[`knowledge/`](knowledge/) as one Markdown file per topic, on three shelves:
methodology, procedures, and a dated landscape of tools. [`llms.txt`](llms.txt) at the
repository root is the index. An agent or a person queries it the same way: fetch the
index, pick a line, fetch that file.

To add or change an entry, edit the file and run
`uv run python scripts/knowledge_index.py`. CI runs the same script with `--check` and
fails when the index is stale or a file breaks the rules in
[`knowledge/README.md`](knowledge/README.md).
```

- [ ] **Step 4: Run everything**

```bash
uv run python scripts/knowledge_index.py --check
uv run pytest -q
uv run ruff check .
uv run mypy src/axial
grep -rn "docs/research" README.md knowledge docs/superpowers/specs/2026-10-07-knowledge-base-layout-design.md
```

Expected: the first four commands succeed. The grep finds only the mentions inside the spec, which describe the migration and are correct history; it must find nothing in `README.md` or `knowledge/`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/research
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -F - <<'EOF'
Retire docs/research in favour of the knowledge base and say so in the README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
```

---

## Self-review against the spec

- Section 1, tree and shelves: Tasks 2 (layout rules), 4 (methodology), 5 and 6 (landscape), 7 (procedures), 8 (deletion of `docs/research`).
- Section 2, frontmatter schema and landscape heading order: Task 1 enforces the schema; Tasks 5 and 6 follow the heading order; `knowledge/README.md` in Task 4 states it. The heading order is not machine-checked; the spec does not ask for that.
- Section 3, index format: Task 3, byte-exact against `EXPECTED_INDEX`.
- Section 4, script functions and CI job: Tasks 1 to 3 define `parse_entry`, `load_tree`, `render_index`, `check`; Task 4 adds the job. `write_index` and `main` are additions the spec's command lines imply.
- Section 5, initial content: Tasks 4 to 7.
- Section 6, error handling: Task 1 (per-file, all violations), Task 2 (sorted, all files), Task 3 (stale and missing index, no write on a broken tree).
- Section 7, testing: every listed case has a test in Tasks 1 to 4, including the real-tree test.
- Section 8, documentation: Task 4 (`knowledge/README.md`), Task 8 (README).
