"""Build and check the knowledge base index.

    python scripts/knowledge_index.py            # rewrite llms.txt
    python scripts/knowledge_index.py --check    # validate; exit 1 on any violation

The tree under knowledge/ is the source of truth. llms.txt is generated from each
file's frontmatter. Every violation is collected before exiting so one run shows the
whole list, and a broken tree never produces an index.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
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
        data, body = split_frontmatter(path.read_text(encoding="utf-8"))
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
        if child.name.startswith("."):
            continue
        rel = child.relative_to(root).as_posix()
        if child.is_dir():
            if child.name not in SHELVES:
                problems.append(f"{rel}: not a shelf; only {', '.join(SHELVES)} are allowed")
                continue
            for item in sorted(child.iterdir()):
                if item.name.startswith("."):
                    continue
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
        if entry.license is not None and entry.shelf != "landscape":
            problems.append(f"{rel}: license: only landscape entries may declare it")
        problems.extend(_link_violations(entry, root))

    if problems:
        raise KnowledgeError(sorted(problems))
    entries.sort(key=lambda e: (SHELVES.index(e.shelf), e.name))
    return entries


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
    if index.read_bytes() != rendered.encode("utf-8"):
        return [f"{INDEX_NAME}: stale; run scripts/knowledge_index.py"]
    return []


def write_index(root: Path) -> list[str]:
    """Rewrite llms.txt, or return the violations and leave it untouched."""
    rendered, problems = _render_or_violations(root)
    if rendered is None:
        return problems
    (root / INDEX_NAME).write_bytes(rendered.encode("utf-8"))
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
