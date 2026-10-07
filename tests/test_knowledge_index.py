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
        (lambda s: s.replace("license: Apache-2.0\n", "license: Apache-2.0\nfoo: bar\n"), "foo: unknown key"),
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


def test_load_tree_rejects_license_outside_landscape(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"procedures/a.md": entry_text("a", "procedures", license="MIT")})
    assert tree_violations(root) == ["knowledge/procedures/a.md: license: only landscape entries may declare it"]


def test_load_tree_ignores_dotfiles(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"landscape/a.md": entry_text("a", "landscape")})
    write(root, "knowledge/.DS_Store", "binary\n")
    write(root, "knowledge/landscape/.hidden", "hidden\n")
    assert [e.name for e in ki.load_tree(root)] == ["a"]


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


def test_check_reports_an_index_with_crlf_line_endings_as_stale(tmp_path: Path) -> None:
    root = three_shelf_repo(tmp_path)
    (root / "llms.txt").write_bytes(EXPECTED_INDEX.replace("\n", "\r\n").encode("utf-8"))
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


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_repository_knowledge_tree_is_valid_and_indexed() -> None:
    assert ki.check(REPO_ROOT) == []
