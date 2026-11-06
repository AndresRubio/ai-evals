"""Check that the tests pin every import rule, by changing one rule at a time.

    uv run python scripts/weaken_rules.py [test file ...] [--list] [--only text] [--workers n]

For a change to the import contracts in pyproject.toml, as CODING_STANDARDS.md asks. A
rule no test pins can be loosened later without anything failing. This copies src, the
test files (tests/test_layers.py when none is named) and pyproject.toml to a temporary
folder once per change, makes the change to the copy's contracts, and runs the tests
there. A change the tests fail on is caught. One they pass on survived: either a test
is missing, or the change makes no difference to what the rules refuse, and the commit
message says which.

The changes, each made alone:

- a layers or forbidden contract refuses no import: it stays in the file, so a test that
  counts the contracts still passes, and every import is on its list of exceptions;
- a layers contract stops being exhaustive, two neighbouring rows swap, two neighbouring
  rows of one module each may import each other, the modules of a shared row ("a | b")
  may import each other, and one module of a shared row moves above or below the rest
  of it;
- a forbidden contract loses one of several source modules or one of several forbidden
  modules, and it loses an import it lets through, which tightens it;
- a contract of any other kind is removed.

It does not try narrower forms of a module, such as one child of a forbidden package.
For a shared row and its neighbour it tries the swap only: the linter has no way to let
the two rows import each other and keep the shared row's modules apart.

--list prints the changes and runs nothing. --only keeps the changes whose line holds
the text. --workers is passed to pytest as -n (auto by default; 0 runs the tests in
one process).

Exits 0 when every change was caught, 1 when one survived, and 2 when it cannot run,
which includes tests that fail on the rules as they stand.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HEADER = "[[tool.importlinter.contracts]]"
Contract = dict[str, Any]


class CannotRun(Exception):
    """The run would prove nothing."""


@dataclass(frozen=True)
class Change:
    """One contract changed or, when `contract` is None, removed."""

    index: int
    rule: str
    what: str
    contract: Contract | None

    @property
    def line(self) -> str:
        return f"{self.rule}: {self.what}"


def blocks(text: str) -> list[tuple[int, int]]:
    """The (start, end) offsets of each contract in pyproject.toml's text. A contract
    ends where the next table starts."""
    found: list[tuple[int, int]] = []
    start: int | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("[") and start is not None:
            found.append((start, offset))
            start = None
        if line.strip() == HEADER:
            start = offset
        offset += len(line)
    if start is not None:
        found.append((start, offset))
    return found


def written(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(written(item) for item in value) + "]"
    raise CannotRun(f"a contract holds a value this script cannot write back: {value!r}")


def block_text(contract: Contract) -> str:
    lines = [f"{key} = {written(value)}" for key, value in contract.items()]
    return "\n".join([HEADER, *lines, "", ""])


def without(contract: Contract, key: str, item: str | None = None) -> Contract:
    """The contract minus a key, or minus one item of the list under it. A list left
    empty goes with its key."""
    changed = dict(contract)
    left = [] if item is None else [other for other in contract[key] if other != item]
    if left:
        changed[key] = left
    else:
        del changed[key]
    return changed


def switched_off(contract: Contract, roots: list[str]) -> Contract:
    """The contract with every import between the root packages on its list of
    exceptions. An exception that matches no import is an error by default, which
    would fail the linter for a reason of its own, so that alert is turned off."""
    everything = [f"{one}.** -> {other}.**" for one in roots for other in roots]
    return {**contract, "ignore_imports": everything, "unmatched_ignore_imports_alerting": "none"}


def layer_changes(contract: Contract) -> Iterator[tuple[str, Contract]]:
    rows: list[str] = contract["layers"]

    def with_rows(new: list[str]) -> Contract:
        return {**contract, "layers": new}

    if contract.get("exhaustive"):
        yield "no longer exhaustive", without(contract, "exhaustive")
    for i, (upper, lower) in enumerate(itertools.pairwise(rows)):
        yield (
            f"{upper!r} and {lower!r} swapped",
            with_rows([*rows[:i], lower, upper, *rows[i + 2 :]]),
        )
        if "|" not in upper + lower and ":" not in upper + lower:
            # A row holds one kind of sibling. Joining a shared row to its neighbour
            # would also let that row's own modules import each other, and a test of
            # theirs would catch the change for the wrong reason, so it is not tried.
            yield (
                f"{upper!r} and {lower!r} may import each other",
                with_rows([*rows[:i], f"{upper} : {lower}", *rows[i + 2 :]]),
            )
    for i, row in enumerate(rows):
        parts = [part.strip() for part in row.split("|")]
        if len(parts) < 2:
            continue
        yield (
            f"the modules of {row!r} may import each other",
            with_rows([*rows[:i], " : ".join(parts), *rows[i + 1 :]]),
        )
        for part in parts:
            rest = " | ".join(other for other in parts if other != part)
            yield f"{part!r} above {rest!r}", with_rows([*rows[:i], part, rest, *rows[i + 1 :]])
            yield f"{part!r} below {rest!r}", with_rows([*rows[:i], rest, part, *rows[i + 1 :]])


def forbidden_changes(contract: Contract) -> Iterator[tuple[str, Contract]]:
    for key, label in (("source_modules", "a source"), ("forbidden_modules", "forbidden")):
        modules: list[str] = contract.get(key, [])
        if len(modules) > 1:
            for module in modules:
                yield f"{module} no longer {label}", without(contract, key, module)
    for entry in contract.get("ignore_imports", []):
        yield f"{entry!r} no longer let through", without(contract, "ignore_imports", entry)


def changes(text: str) -> list[Change]:
    """Every change to try, in the order of the contracts. Two changes that leave a
    contract reading the same count once."""
    try:
        settings = tomllib.loads(text).get("tool", {}).get("importlinter", {})
    except tomllib.TOMLDecodeError as error:
        raise CannotRun(f"pyproject.toml does not parse: {error}") from error
    roots: list[str] = settings.get("root_packages") or [settings.get("root_package", "")]
    found: list[Change] = []
    for index, (start, end) in enumerate(blocks(text)):
        contract: Contract = tomllib.loads(text[start:end])["tool"]["importlinter"]["contracts"][0]
        rule = str(contract.get("name", f"contract {index + 1}"))
        kind = contract.get("type")
        each: Iterator[tuple[str, Contract]]
        if kind == "layers":
            each = layer_changes(contract)
        elif kind == "forbidden":
            each = forbidden_changes(contract)
        else:
            found.append(Change(index, rule, "removed", None))
            continue
        seen = {block_text(contract)}
        for what, changed in [("no import refused", switched_off(contract, roots)), *each]:
            if block_text(changed) not in seen:
                seen.add(block_text(changed))
                found.append(Change(index, rule, what, changed))
    return found


def contracts(text: str) -> list[Contract]:
    found: list[Contract] = tomllib.loads(text)["tool"]["importlinter"].get("contracts", [])
    return found


def applied(text: str, change: Change) -> str:
    """pyproject.toml's text with one change made. The result is read back, since a file
    the linter cannot load would fail every test and pass for a caught change."""
    start, end = blocks(text)[change.index]
    block = "" if change.contract is None else block_text(change.contract)
    changed = text[:start] + block + text[end:]
    expected = contracts(text)
    expected[change.index : change.index + 1] = [] if change.contract is None else [change.contract]
    try:
        if contracts(changed) != expected:
            raise CannotRun(f"could not write this change back to pyproject.toml: {change.line}")
    except (tomllib.TOMLDecodeError, KeyError) as error:
        raise CannotRun(
            f"{change.line} left a pyproject.toml that does not parse: {error}"
        ) from error
    return changed


def tests_pass(pyproject: str, tests: list[str], workers: str) -> tuple[bool, str]:
    """Whether the tests pass against a copy of the repository with this pyproject.toml,
    and the last line pytest printed."""
    with tempfile.TemporaryDirectory(prefix="weaken-rules-") as folder:
        root = Path(folder)
        shutil.copytree(
            "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
        )
        for test in tests:
            for path in (Path(test), Path(test).parent / "conftest.py"):
                if path.is_file():
                    (root / path).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(path, root / path)
        (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        parallel = [] if workers == "0" else ["-n", workers]
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                *parallel,
                *tests,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        lines = (done.stdout.strip() or done.stderr.strip()).splitlines()
        # Every test runs, so pytest exits 1 for failed tests alone. Anything else, such as
        # 5 for no test collected, is a run that proves nothing either way.
        if done.returncode not in (0, 1):
            raise CannotRun(
                f"pytest exited {done.returncode}: {lines[-1] if lines else 'no output'}"
            )
        return done.returncode == 0, lines[-1] if lines else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("tests", nargs="*", default=["tests/test_layers.py"])
    parser.add_argument("--list", action="store_true", help="print the changes and stop")
    parser.add_argument("--only", metavar="text", help="keep the changes whose line holds this")
    parser.add_argument("--workers", default="auto", help="pytest's -n (auto); 0 for one process")
    args = parser.parse_args()
    try:
        if not Path("pyproject.toml").is_file() or not Path("src").is_dir():
            raise CannotRun(
                "run this from the top of the repository, where pyproject.toml and src are"
            )
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        todo = [change for change in changes(text) if args.only is None or args.only in change.line]
        if not todo:
            raise CannotRun(
                "no change to try: pyproject.toml has no import contract, or --only matched none"
            )
        if args.list:
            print("\n".join(change.line for change in todo))
            return 0
        missing = [test for test in args.tests if not Path(test).is_file()]
        if missing:
            raise CannotRun(f"no such test file: {', '.join(missing)}")
        outside = [
            test for test in args.tests if Path(test).is_absolute() or ".." in Path(test).parts
        ]
        if outside:
            raise CannotRun(
                f"name a test file by its path inside the repository: {', '.join(outside)}"
            )
        passed, last = tests_pass(text, args.tests, args.workers)
        if not passed:
            raise CannotRun(f"the tests fail on the rules as they stand: {last}")
        survived = 0
        for change in todo:
            passed, _ = tests_pass(applied(text, change), args.tests, args.workers)
            survived += passed
            print(f"{'SURVIVED' if passed else 'caught  '}  {change.line}", flush=True)
    except CannotRun as error:
        print(f"weaken_rules: {error}", file=sys.stderr)
        return 2
    print(f"{len(todo)} changes tried, {survived} survived")
    return 1 if survived else 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:  # noqa: BLE001 - whatever went wrong, the answer is exit 2
        # Python's own exit status for a crash is 1, which here would read as
        # "a change survived".
        traceback.print_exc()
        code = 2
    sys.exit(code)
