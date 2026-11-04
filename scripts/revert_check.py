"""Check that a commit's tests fail without its source change, as CODING_STANDARDS.md asks.

    uv run python scripts/revert_check.py <commit> [test id ...] [--guard id]... [--at rev]

In a temporary checkout of the commit, this runs the tests with the change in place, then
again with the commit's changes under src reversed. The tests are the ones the commit's diff
adds or changes, unless you name others. Your working tree stays out of it, so an interrupted
run leaves src as it was, and a later commit that uses the reversed lines cannot make every
test fail for that reason alone.

--guard names one test that may pass with the change reversed, so repeat it for more. --at
runs in a checkout of a later commit, for tests that arrived after the change.

Exits 0 when at least one test failed with the change reversed and every other test failed
too or is a named guard that passed. Exits 1 when that does not hold. That includes a test
that did not run, a test that does not pass with the change in place, as its failing would
then prove nothing, and a test that fails only on a missing import or module attribute, as
that usually shows a name the commit adds exists and nothing more. Without --at, it also
includes a test the commit changes that did not run because you named others. Exits 2 when
the check cannot run.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
SEVERITY = ["passed", "skipped", "failed"]
# What a test fails on when it reaches for a name that is not there: an import, or a
# module's attribute read directly or through monkeypatch.setattr. A missing attribute on a
# value is an ordinary failure, and so is a new keyword, which cannot be told from one.
MISSING_NAME_RE = re.compile(
    r"(ImportError|ModuleNotFoundError):|AttributeError: (module '|'module' object at |<module ')"
)

# Loaded into the nested pytest, so an outcome comes with its whole id and from every phase
# of the test, which pytest's summary lines do not give.
PLUGIN = """\
import json
import os

log = open(os.environ["REVERT_CHECK_LOG"], "a", encoding="utf-8")


def write(**row):
    log.write(json.dumps(row) + "\\n")
    log.flush()


def pytest_runtest_logreport(report):
    # A test that kills pytest during its call leaves a passed setup, which is no pass.
    if report.when == "call" or not report.passed:
        crash = getattr(report.longrepr, "reprcrash", None)
        write(test=report.nodeid, outcome=report.outcome, reason=crash.message if crash else "")


def pytest_collectreport(report):
    if report.failed:
        write(error=str(report.longrepr))
"""


class CannotCheck(Exception):
    """The check could not run, for the reason in the message."""


def git(*args: str, cwd: Path | None = None, stdin: str | None = None) -> str:
    """Run git, in bytes so that a patch keeps its line endings."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        input=None if stdin is None else stdin.encode("utf-8", "surrogateescape"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CannotCheck(f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode("utf-8", "surrogateescape")


def diff_tree(commit: str, *options: str, path: str) -> str:
    return git("diff-tree", "--no-commit-id", "-r", *options, commit, "--", path)


def first_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])


def is_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any("fixture" in ast.unparse(decorator) for decorator in node.decorator_list)


def touched_in(path: str, source: str, lines: set[int]) -> tuple[list[str], list[int]]:
    """The test functions in `source` holding one of `lines`, decorators included, and the
    lines that fall outside any test function, with imports, docstrings, test class headers,
    blank lines and comments aside."""
    try:
        # As bytes, so that Python reads the file's own encoding line or BOM.
        module = ast.parse(source.encode("utf-8", "surrogateescape"))
    except (SyntaxError, ValueError) as exc:
        raise CannotCheck(f"could not parse {path}: {exc}") from exc

    found: list[str] = []
    covered: set[int] = set()

    def visit(body: Iterable[ast.stmt], prefix: str) -> None:
        for node in body:
            last = (node.end_lineno or node.lineno) + 1
            is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            if is_docstring or isinstance(node, ast.Import | ast.ImportFrom):
                covered.update(range(node.lineno, last))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name.startswith("test") and not is_fixture(node):
                    covered.update(range(first_line(node), last))
                    if lines.intersection(range(first_line(node), last)):
                        found.append(f"{prefix}::{node.name}")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                covered.add(node.lineno)
                visit(node.body, f"{prefix}::{node.name}")

    visit(module.body, path)
    text = source.split("\n")
    outside = [
        number
        for number in sorted(lines - covered)
        if text[number - 1].strip() and not text[number - 1].lstrip().startswith("#")
    ]
    return found, outside


def touched_tests(commit: str) -> tuple[list[str], list[str]]:
    """The test functions `commit` adds or changes, and what else it changes under tests: a
    `path:line` for a test file changed outside a test function, or the path of any other file.
    Deleted lines touch nothing."""
    found: list[str] = []
    outside: list[str] = []
    paths = diff_tree(commit, "-z", "--name-only", "--diff-filter=d", path="tests")
    for path in filter(None, paths.split("\0")):
        name = Path(path).name
        if not (name.startswith("test_") and name.endswith(".py")):
            outside.append(path)
            continue
        lines: set[int] = set()
        for start, count in HUNK_RE.findall(diff_tree(commit, "-p", "-U0", path=path)):
            lines.update(range(int(start), int(start) + int(count or 1)))
        tests, others = touched_in(path, git("show", f"{commit}:{path}"), lines)
        found += tests
        if others:
            outside.append(f"{path}:{others[0]}")
    return found, outside


def selects(selector: str, node_id: str) -> bool:
    """Whether a test id, file or directory on the command line covers `node_id`, whose
    parametrized cases it includes."""
    path, separator, rest = selector.partition("::")
    path = os.path.normpath(path)
    selector = f"{path}{separator}{rest}"
    return (
        path == "."
        or node_id == selector
        or node_id.startswith((f"{selector}[", f"{selector}::", f"{selector}/"))
    )


@dataclass
class Run:
    outcomes: dict[str, str]
    """Each test that ran and its worst outcome: passed, skipped or failed."""
    reasons: dict[str, str]
    """What each failed test failed on, the first time it failed."""
    not_run: dict[str, str]
    """Each test file pytest could not run, with the reason."""


def reason_in(output: str) -> str:
    """The line of pytest's output that says what went wrong."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("E ")]
    if errors:
        return errors[-1].removeprefix("E").strip()
    usage = [line for line in lines if line.startswith("ERROR:")]
    return usage[0] if usage else (lines[-1] if lines else "pytest printed nothing")


def run_tests(checkout: Path, tests: Iterable[str], plugin_dir: Path) -> Run:
    """Run `tests` in `checkout`, a file at a time, as a file that does not import stops
    pytest before it runs anything else on its command line."""
    by_file: dict[str, list[str]] = {}
    for test in tests:
        by_file.setdefault(test.split("::")[0], []).append(test)

    log = plugin_dir / "outcomes.jsonl"
    python_path = os.pathsep.join(filter(None, [str(plugin_dir), os.environ.get("PYTHONPATH")]))
    env = {
        **os.environ,
        "PYTHONPATH": python_path,
        "REVERT_CHECK_LOG": str(log),
        # A pyc counts as current for a source of the same size and second, so a run could
        # use the baseline's bytecode for the reversed source.
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    run = Run({}, {}, {})
    for path, ids in by_file.items():
        log.unlink(missing_ok=True)
        try:
            result = subprocess.run(
                ["uv", "run", "-q", "pytest", "-q", "--color=no", "-p", "revert_check_plugin", *ids],
                cwd=checkout,
                env=env,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            raise CannotCheck(f"could not run uv: {exc}") from exc

        lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        rows = [json.loads(line) for line in lines]
        for row in rows:
            if "test" in row:
                seen = run.outcomes.get(row["test"], "passed")
                run.outcomes[row["test"]] = max(seen, row["outcome"], key=SEVERITY.index)
                if row["outcome"] == "failed":
                    run.reasons.setdefault(row["test"], row["reason"].split("\n")[0])
        errors = [row["error"] for row in rows if "error" in row]
        # No row at all means pytest never got going, whatever uv exited with.
        if errors or not rows or result.returncode not in (0, 1):
            run.not_run[path] = reason_in(errors[-1] if errors else result.stdout + result.stderr)
    return run


def section(title: str, items: Iterable[str]) -> None:
    lines = sorted(items)
    if lines:
        print(f"\n{title}")
        for line in lines:
            print(f"  {line}")


def check(commit: str, named: list[str], guards: list[str], at: str | None) -> int:
    """Run the check and print what happened. The return value is the exit status."""
    if not all(named):
        raise CannotCheck("a test name is empty")
    if any(os.path.normpath(guard or ".") == "." for guard in guards):
        raise CannotCheck("--guard names a test or a file, and this one covers everything")
    try:
        git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
        parents = git("rev-list", "--parents", "-n", "1", f"{commit}^{{commit}}").split()
    except CannotCheck as exc:
        raise CannotCheck(f"{commit} is not a commit") from exc
    if len(parents) > 2:
        raise CannotCheck(f"{commit} is a merge; name one of the commits it brought in")
    patch = diff_tree(commit, "-p", "--binary", path="src")
    if not patch:
        raise CannotCheck(f"{commit} changes nothing under src")
    touched, outside = touched_tests(commit)
    tests = named or touched
    if not tests:
        raise CannotCheck(f"{commit} adds or changes no test; name the tests to run")
    selected = list(dict.fromkeys([*tests, *guards]))
    left_out = [test for test in touched if not any(selects(name, test) for name in selected)]
    place = at or commit

    with tempfile.TemporaryDirectory(prefix="revert-check-") as parent:
        plugin_dir = Path(parent)
        (plugin_dir / "revert_check_plugin.py").write_text(PLUGIN, encoding="utf-8")
        checkout = plugin_dir / "checkout"
        git("worktree", "add", "-q", "--detach", str(checkout), place)
        try:
            try:
                git("apply", "-R", "--check", cwd=checkout, stdin=patch)
            except CannotCheck as exc:
                raise CannotCheck(
                    f"{commit}'s source change does not reverse cleanly at {place}"
                ) from exc

            baseline = run_tests(checkout, selected, plugin_dir)
            if baseline.not_run:
                detail = "\n".join(f"  {path}: {why}" for path, why in baseline.not_run.items())
                raise CannotCheck(f"pytest could not run these at {place}:\n{detail}")
            not_passing = [test for test, seen in baseline.outcomes.items() if seen != "passed"]
            not_passing += [
                test
                for test in selected
                if not any(selects(test, node_id) for node_id in baseline.outcomes)
            ]
            if not_passing:
                section("These do not pass even with the change in place:", not_passing)
                return 1

            # A file the first run left could fail a test, which would look like the reversal.
            git("checkout", "-q", "--", ".", cwd=checkout)
            git("clean", "-fdxq", "-e", ".venv", cwd=checkout)
            git("apply", "-R", cwd=checkout, stdin=patch)
            reverted = run_tests(checkout, selected, plugin_dir)
        finally:
            git("worktree", "remove", "--force", str(checkout))

    def with_reason(test: str) -> str:
        return f"{test}\n    {reverted.reasons[test]}"

    failed: list[str] = []
    missing_name: list[str] = []
    guarded: list[str] = []
    unnamed: list[str] = []
    did_not_run: list[str] = []
    for node_id in baseline.outcomes:
        seen = reverted.outcomes.get(node_id, "not run")
        if seen == "failed" and MISSING_NAME_RE.match(reverted.reasons[node_id]):
            missing_name.append(node_id)
        elif seen == "failed":
            failed.append(node_id)
        elif seen == "passed" and any(selects(guard, node_id) for guard in guards):
            guarded.append(node_id)
        elif seen == "passed":
            unnamed.append(node_id)
        elif not any(selects(path, node_id) for path in reverted.not_run):
            did_not_run.append(node_id)

    print(f"Reversed {commit}'s changes under src at {place}.")
    section("These fail with the change reversed:", map(with_reason, failed))
    section(
        "These fail only on a missing import or module attribute, which usually shows that a "
        "name the commit adds exists and nothing more:",
        map(with_reason, missing_name),
    )
    section("These pass with the change reversed, and are named guards:", guarded)
    section("These did not run with the change reversed:", did_not_run)
    section(
        "These files did not run with the change reversed, so their tests show nothing about it:",
        (f"{path}: {why}" for path, why in reverted.not_run.items()),
    )
    section("These pass with the change reversed, and are not named with --guard:", unnamed)
    if {test.split("[")[0] for test in unnamed} & {test.split("[")[0] for test in failed}:
        print(
            "\nSome of these are cases of a test with other cases that fail. If the commit only "
            "added those\nother cases, the ones here are older than it: name those cases with "
            "--guard."
        )
    section(
        "The commit changes these tests too, and they did not run because you named others:",
        left_out,
    )
    section(
        "These changed in test files outside a test function, and the check cannot tell which "
        "tests they affect:",
        outside,
    )
    if not failed and not missing_name:
        print("\nNothing failed with the change reversed.")
    proved_nothing = unnamed or did_not_run or missing_name or reverted.not_run
    # With --at, leaving the commit's own tests out to name later ones is the point.
    narrowed = left_out and at is None
    return 1 if proved_nothing or narrowed or not failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("commit")
    parser.add_argument("tests", nargs="*", help="test ids; by default the ones the commit touches")
    parser.add_argument(
        "--guard",
        action="append",
        default=[],
        metavar="ID",
        help="a test that may pass with the change reversed; repeat for more",
    )
    parser.add_argument("--at", metavar="REV", help="run in a checkout of REV instead of the commit")
    args = parser.parse_intermixed_args(argv)

    # A git hook sets GIT_INDEX_FILE, which `git worktree add` would write over, and
    # PYTEST_ADDOPTS could stop a run early. Being told to stop exits through the checkout's
    # removal.
    os.environ.pop("GIT_INDEX_FILE", None)
    os.environ.pop("PYTEST_ADDOPTS", None)
    for stop in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(stop, lambda number, _: sys.exit(128 + number))
    # A test id or a failure message can hold what the terminal cannot print.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="backslashreplace")

    try:
        os.chdir(git("rev-parse", "--show-toplevel").strip())
        return check(args.commit, args.tests, args.guard, args.at)
    except CannotCheck as exc:
        print(f"revert_check: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        # This says nothing about the commit, and a traceback's exit 1 would say it failed.
        print(f"revert_check: could not finish: {exc!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
