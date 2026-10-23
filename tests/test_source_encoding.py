"""Every file `src/` and `scripts/` open, read or write as text names its encoding.

Left out, the encoding is the locale's, which is ASCII under a bare C locale, so a
hand-edited UTF-8 file would be refused on one machine and read on the next. Ruff's rule
for this (PLW1514) runs only in preview mode and does not look at `os.fdopen`, so the
check reads the source itself.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted([*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py")])


def _positions(func: ast.expr) -> tuple[int | None, int] | None:
    """Where a call that opens, reads or writes text takes its mode and its encoding when
    they are passed by position, or None if `func` is no such call."""
    match func:
        case ast.Attribute(ast.Name("os"), "open"):
            return None  # a file descriptor, never text
        case ast.Name("open") | ast.Attribute(ast.Name("io"), "open"):
            return 1, 3
        case ast.Attribute(ast.Name("os"), "fdopen"):
            return 1, 3
        case ast.Attribute(attr="open"):
            return 0, 2  # Path.open
        case ast.Attribute(attr="read_text"):
            return None, 0
        case ast.Attribute(attr="write_text"):
            return None, 1
    return None


def unpinned(source: str) -> list[int]:
    """The lines of `source` with a call that opens, reads or writes text and names no
    encoding."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or (positions := _positions(node.func)) is None:
            continue
        mode_at, encoding_at = positions
        keywords = {k.arg: k.value for k in node.keywords}
        if "encoding" in keywords or encoding_at < len(node.args):
            continue
        mode = keywords.get("mode")
        if mode is None and mode_at is not None and mode_at < len(node.args):
            mode = node.args[mode_at]
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value:
            continue
        lines.append(node.lineno)
    return lines


@pytest.mark.parametrize(
    "call",
    [
        'open("labels.jsonl")',
        'open(path, "w")',
        "io.open(path)",
        'os.fdopen(fd, "w")',
        "path.open()",
        'path.open("a")',
        "path.open(mode=mode)",
        "path.read_text()",
        "path.write_text(data)",
        'path.write_text(data, errors="strict")',
    ],
)
def test_a_text_call_with_no_encoding_is_found(call: str) -> None:
    assert unpinned(call) == [1]


@pytest.mark.parametrize(
    "call",
    [
        'open(path, encoding="utf-8")',
        'open(path, "r", -1, "utf-8")',
        'open(path, "rb")',
        'os.fdopen(fd, "wb")',
        'path.open("rb")',
        'path.open(mode="ab")',
        'path.open("w", -1, "utf-8")',
        'path.read_text("utf-8")',
        'path.write_text(data, "utf-8")',
        "os.open(path, os.O_RDONLY)",
        "path.read_bytes()",
    ],
)
def test_a_binary_call_or_one_naming_its_encoding_is_not(call: str) -> None:
    assert unpinned(call) == []


def test_every_text_file_the_package_and_scripts_touch_names_its_encoding() -> None:
    assert SOURCES, "no source files found"
    found = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in SOURCES
        for line in unpinned(path.read_text(encoding="utf-8"))
    ]
    assert found == [], "pass encoding=\"utf-8\" to each of these: " + ", ".join(found)
