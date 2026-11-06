"""List what differs between the code at a base revision and the code now, wherever it moved.

    uv run python scripts/moved_code.py <base> [path ...] [--rename old=new]... [--diff]

For a commit that moves code and says nothing changed, as CODING_STANDARDS.md asks. Every
top-level statement in the Python files under the paths (src when none is named) is taken
whole: the comment lines above it, its decorators, its docstring and its body. It is filed
under the name it defines, and a module's docstring under "<module docstring>". A name is
compared in its own file first, then across all the files, so a function that moved to
another module, or a module split in two, compares equal as long as its text stayed the same.

The base is read from git and the other side from the working tree, tracked or not. A
top-level import is left out, since a move rewrites those, and only its comments are
compared. An import inside a function is part of that function and shows as a change.

Prints one line per name that needs reading, with the files that hold it:

    changed  its text differs
    gone     the base has it and the working tree does not
    new      the working tree has it and the base does not
    moved    its text is the same, and its new file may give it another meaning: it uses
             a name that is another statement there, or one its old file left to an import
             or a builtin and its new file defines; it reads __file__, __name__ or
             __package__; it has a relative import; or it declares a global

--diff adds the differing lines. --rename applies to the base a rename the commit makes
on purpose, so the renamed name compares equal and the rest of it is still checked. It
renames bare uses of the name in code and says how many, and a parameter or a local of
the same name is a bare use too, so pick names nothing else uses. An attribute
(`x.name`), a keyword (`f(name=1)`), a comment and a string keep the old name and show
as changed where the commit renamed them too.

What it cannot see, and the review still reads: a name the moved code gets from a
top-level import, which may be another thing in the new module; the order of
statements; code that moved into or out of a class or function; and files that are
neither .py nor .pyi.

Exits 0 when nothing is printed, 1 when something is, and 2 when it cannot run.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import subprocess
import sys
import tokenize
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

DOCSTRING = "<module docstring>"
# How alike two texts under one name must be to be shown as one changed piece.
NEAR = 0.5


class CannotRun(Exception):
    """The comparison has nothing sound to read."""


@dataclass(frozen=True)
class Piece:
    """One top-level statement with the comment lines above it."""

    name: str
    text: str
    path: str


@dataclass(frozen=True)
class Difference:
    """A name to read. `before` is None for a new name and `after` for one gone. `note`
    is set for a piece whose text is the same and whose new file changes what it means."""

    name: str
    before: Piece | None
    after: Piece | None
    note: str = ""

    @property
    def kind(self) -> str:
        if self.note:
            return "moved"
        if self.before is None:
            return "new"
        return "gone" if self.after is None else "changed"


def name_of(node: ast.stmt, index: int, first_line: str) -> str | None:
    """What a statement is filed under, or None for a top-level import."""
    if isinstance(node, ast.Import | ast.ImportFrom):
        return None
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if names:
            return ",".join(names)
    is_docstring = (
        index == 0
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
    # A statement that defines no name, such as `if TYPE_CHECKING:`, goes by its first line.
    return DOCSTRING if is_docstring else f"<{first_line.strip()}>"


def pieces(path: str, text: str) -> list[Piece]:
    """The top-level statements of one file. The comment lines between two statements
    belong to the second, and those after the last are filed on their own."""
    try:
        body = ast.parse(text).body
    except SyntaxError as error:
        raise CannotRun(f"{path} does not parse: {error.msg} (line {error.lineno})") from error
    lines = text.splitlines()
    found: list[Piece] = []
    done = 0
    for index, node in enumerate(body):
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        end = node.end_lineno or node.lineno
        comments = [line for line in lines[done : start - 1] if line.strip()]
        name = name_of(node, index, lines[start - 1])
        if name is None:
            # An import is left out, and what is written about it is kept.
            comments += [
                token.string
                for token in tokens("\n".join(lines[start - 1 : end]))
                if token.type == tokenize.COMMENT
            ]
            if comments:
                found.append(Piece("<comment at an import>", "\n".join(comments), path))
        else:
            found.append(Piece(name, "\n".join([*comments, *lines[start - 1 : end]]), path))
        done = end
    trailing = [line for line in lines[done:] if line.strip()]
    if trailing:
        found.append(Piece("<comment at the end>", "\n".join(trailing), path))
    return found


def tokens(text: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(text + "\n").readline))
    except (tokenize.TokenError, SyntaxError) as error:
        raise CannotRun(f"cannot read the tokens of {text.splitlines()[0]!r}: {error}") from error


def names_in(text: str) -> set[str]:
    """The names code uses bare. `error.__name__` reads an attribute, which the module
    it sits in does not decide."""
    code = [token for token in tokens(text) if token.type != tokenize.COMMENT]
    return {
        token.string
        for index, token in enumerate(code)
        if token.type == tokenize.NAME and (index == 0 or code[index - 1].string != ".")
    }


def renamed(text: str, renames: Mapping[str, str], counts: dict[str, int]) -> str:
    """`text` with each renamed name replaced where code uses it bare, counted in
    `counts`. An attribute, a keyword, a comment and a string keep the name, so one
    the commit renamed as well still shows."""
    if not renames:
        return text
    lines = text.splitlines(keepends=True)
    code = [
        token
        for token in tokens(text)
        if token.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE)
    ]
    depth = 0
    uses: list[tokenize.TokenInfo] = []
    for index, token in enumerate(code):
        if token.type == tokenize.OP:
            depth += token.string in "([{"
            depth -= token.string in ")]}"
        if token.type != tokenize.NAME or token.string not in renames:
            continue
        is_attribute = index > 0 and code[index - 1].string == "."
        is_keyword = depth > 0 and index + 1 < len(code) and code[index + 1].string == "="
        if not is_attribute and not is_keyword:
            uses.append(token)
    for token in reversed(uses):
        row, column = token.start
        line = lines[row - 1]
        lines[row - 1] = line[:column] + renames[token.string] + line[token.end[1] :]
        counts[token.string] = counts.get(token.string, 0) + 1
    return "".join(lines)


def by_name(
    files: Mapping[str, str], renames: Mapping[str, str], counts: dict[str, int]
) -> dict[str, list[Piece]]:
    filed: dict[str, list[Piece]] = defaultdict(list)
    for path in sorted(files):
        for piece in pieces(path, files[path]):
            name = renames.get(piece.name, piece.name)
            filed[name].append(Piece(name, renamed(piece.text, renames, counts), piece.path))
    return filed


def paired(
    before: list[Piece], after: list[Piece], same: Callable[[Piece, Piece], bool]
) -> list[tuple[Piece, Piece]]:
    """Takes out of both lists each piece of `before` with the first of `after` that is
    the same in the sense asked, and returns the pairs."""
    pairs: list[tuple[Piece, Piece]] = []
    for piece in list(before):
        twin = next((other for other in after if same(piece, other)), None)
        if twin is not None:
            before.remove(piece)
            after.remove(twin)
            pairs.append((piece, twin))
    return pairs


def texts_in(filed: Mapping[str, list[Piece]]) -> dict[tuple[str, str], set[str]]:
    """The texts each file has under each name."""
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for name, group in filed.items():
        for piece in group:
            found[piece.path, name].add(piece.text)
    return found


MODULE_NAMES = {"__file__", "__name__", "__package__", "__spec__", "__path__"}


def means_something_else(
    old: Piece,
    new: Piece,
    was: Mapping[tuple[str, str], set[str]],
    now: Mapping[tuple[str, str], set[str]],
    listed: set[tuple[str, str, str]],
) -> str:
    """Why a piece that moved with its text intact needs reading, or "" when nothing
    here says so. A name already listed as changed between the two files is read
    there, and is not repeated for each piece that uses it."""
    used = names_in(new.text) - {new.name}
    notes: list[str] = []
    for name in sorted(used):
        there = now.get((new.path, name))
        before = was.get((old.path, name))
        if there is None or (name, old.path, new.path) in listed:
            continue
        if before is None:
            # The old file got the name from an import or from the builtins.
            notes.append(f"{name} is a statement there")
        elif before != there:
            notes.append(f"{name} is another statement there")
    notes += [f"reads {name}" for name in sorted(used & MODULE_NAMES)]
    for node in ast.walk(ast.parse(new.text)):
        if isinstance(node, ast.ImportFrom) and node.level:
            notes.append(f"imports from {'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Global):
            notes.append(f"declares global {', '.join(node.names)}")
    return ", ".join(dict.fromkeys(notes))


def differences(
    before: Mapping[str, str], after: Mapping[str, str], renames: Mapping[str, str]
) -> tuple[int, list[Difference], dict[str, int]]:
    """How many names were compared, each one to read, and how many uses each rename
    replaced. `before` and `after` map a file's path to its text; `renames` is applied
    to `before` alone."""
    counts: dict[str, int] = {}
    old, new = by_name(before, renames, counts), by_name(after, {}, {})
    found: list[Difference] = []
    moved: list[tuple[Piece, Piece]] = []
    for name in sorted(old.keys() | new.keys()):
        left, right = list(old.get(name, [])), list(new.get(name, []))
        # In its own file first: two modules that swap a constant of one name each
        # changed, though both texts are still somewhere.
        paired(left, right, lambda one, other: one.path == other.path and one.text == other.text)
        for one, other in paired(left, right, lambda one, other: one.path == other.path):
            found.append(Difference(name, one, other))
        moved += paired(left, right, lambda one, other: one.text == other.text)
        # What is left on each side is paired with its nearest text, and a piece with
        # nothing near it is gone or new.
        for piece in left:
            near = max(right, key=lambda other: likeness(piece, other), default=None)
            if near is not None and likeness(piece, near) >= NEAR:
                right.remove(near)
                found.append(Difference(name, piece, near))
            else:
                found.append(Difference(name, piece, None))
        found.extend(Difference(name, None, piece) for piece in right)
    was, now = texts_in(old), texts_in(new)
    listed = {(d.name, d.before.path, d.after.path) for d in found if d.before and d.after}
    for one, other in moved:
        note = means_something_else(one, other, was, now, listed)
        if note:
            found.append(Difference(one.name, one, other, note))
    found.sort(key=lambda difference: difference.name)
    return len(old.keys() | new.keys()), found, counts


def likeness(one: Piece, other: Piece) -> float:
    return difflib.SequenceMatcher(None, one.text, other.text).ratio()


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise CannotRun(done.stderr.strip() or f"git {' '.join(args)} failed")
    return done.stdout


def python_files(listing: str) -> list[str]:
    return [path for path in listing.split("\0") if path.endswith((".py", ".pyi"))]


def at_base(base: str, paths: list[str]) -> dict[str, str]:
    git("rev-parse", "--verify", "--quiet", f"{base}^{{tree}}")
    listed = python_files(git("ls-tree", "-r", "-z", "--name-only", base, "--", *paths))
    return {path: git("show", f"{base}:{path}") for path in listed}


def in_working_tree(paths: list[str]) -> dict[str, str]:
    listed = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths)
    files = (Path(path) for path in python_files(listed))
    return {str(path): path.read_text(encoding="utf-8") for path in files if path.is_file()}


def report(difference: Difference, with_lines: bool) -> str:
    before, after = difference.before, difference.after
    where = " -> ".join(dict.fromkeys(piece.path for piece in (before, after) if piece))
    note = f"  ({difference.note})" if difference.note else ""
    out = [f"{difference.kind:<8}{difference.name}  {where}{note}"]
    if with_lines and not difference.note:
        out.extend(
            difflib.unified_diff(
                before.text.splitlines() if before else [],
                after.text.splitlines() if after else [],
                before.path if before else "/dev/null",
                after.path if after else "/dev/null",
                lineterm="",
            )
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("base", help="the revision the code moved from")
    parser.add_argument("paths", nargs="*", default=["src"], help="folders or files (src)")
    parser.add_argument("--rename", action="append", default=[], metavar="old=new")
    parser.add_argument("--diff", action="store_true", help="print the differing lines too")
    args = parser.parse_args()
    renames = dict(pair.split("=", 1) for pair in args.rename if "=" in pair)
    if len(renames) != len(args.rename):
        parser.error("--rename takes old=new")
    try:
        # Paths are read from the top of the repository on both sides.
        top = git("rev-parse", "--show-toplevel").strip()
        if Path(top).resolve() != Path.cwd().resolve():
            raise CannotRun(f"run this from the top of the repository, {top}")
        compared, found, counts = differences(
            at_base(args.base, args.paths), in_working_tree(args.paths), renames
        )
    except CannotRun as error:
        print(f"moved_code: {error}", file=sys.stderr)
        return 2
    for difference in found:
        print(report(difference, args.diff))
    for old, new in renames.items():
        uses = counts.get(old, 0)
        print(f"--rename {old}={new}: {uses} use{'' if uses == 1 else 's'} renamed at the base")
    print(f"{compared} names compared, {len(found)} differ")
    return 1 if found else 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:  # noqa: BLE001 - whatever went wrong, the answer is exit 2
        # A crash is no answer: Python's own exit status for it is 1, which here
        # would read as "something differs".
        traceback.print_exc()
        code = 2
    sys.exit(code)
