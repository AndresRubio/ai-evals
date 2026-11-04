"""Tests for scripts/revert_check.py, run against a small temporary repo.

A fake `uv` on PATH hands `uv run -q pytest ...` to this interpreter's pytest, so the nested
run needs no environment of its own, and the script's temporary checkouts run real tests.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "revert_check.py"

FAKE_UV = f"""#!/bin/sh
if [ -n "${{UV_HANGS:-}}" ]; then
  touch "$UV_HANGS"
  exec sleep 30
fi
if [ -n "${{UV_BROKEN:-}}" ]; then
  echo "$UV_BROKEN" >&2
  exit 1
fi
shift 3
exec "{sys.executable}" -m pytest "$@"
"""

BASE_SOURCE = """def accepts(text):
    return True
"""

CHANGED_SOURCE = """def accepts(text):
    return text != ""
"""

BASE_TESTS = """from app import accepts


def test_older_one():
    assert accepts("x")
"""

REFUSAL_TEST = """

def test_refuses_empty():
    assert not accepts("")
"""

GUARD_TEST = """

def test_still_accepts_a_space():
    assert accepts(" ")
"""

REFUSAL = "tests/test_app.py::test_refuses_empty"
GUARD = "tests/test_app.py::test_still_accepts_a_space"
OLDER = "tests/test_app.py::test_older_one"

FAILS = "These fail with the change reversed:"
GUARDS = "These pass with the change reversed, and are named guards:"
UNNAMED = "These pass with the change reversed, and are not named with --guard:"
DID_NOT_RUN = "These did not run with the change reversed:"
NOT_PASSING = "These do not pass even with the change in place:"
UNIMPORTED = "These files did not run with the change reversed"
OUTSIDE = "These changed in test files outside a test function"
MISSING_NAME = "These fail only on a missing import or module attribute"
LEFT_OUT = "The commit changes these tests too"
NOTHING_FAILED = "Nothing failed with the change reversed."


def listed_under(result: subprocess.CompletedProcess[str], heading: str) -> list[str]:
    """The items the script printed under the heading that starts with `heading`, without the
    reason a failed test's item has on a further indented line."""
    lines = result.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(heading):
            found = []
            for item in lines[index + 1 :]:
                if not item.startswith("  "):
                    break
                if not item.startswith("    "):
                    found.append(item.strip())
            return found
    return []


@dataclass
class Repo:
    root: Path
    env: dict[str, str]

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.root, env=self.env, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def write(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def append(self, name: str, text: str) -> None:
        path = self.root / name
        path.write_text(path.read_text() + text)

    def commit(self, subject: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", subject)
        return self.git("rev-parse", "--short", "HEAD")

    def revert_check(
        self, *args: str, cwd: Path | None = None, **env: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=cwd or self.root,
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=False,
        )

    def worktrees(self) -> int:
        return len(self.git("worktree", "list").splitlines())


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    """A repo whose one source file accepts everything and whose one test says so."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(FAKE_UV)
    fake_uv.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env |= {
        "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    root = tmp_path / "repo"
    root.mkdir()
    made = Repo(root, env)
    made.git("init", "-q", "-b", "main")
    made.git("config", "user.name", "Test")
    made.git("config", "user.email", "test@example.com")
    made.write("pyproject.toml", '[tool.pytest.ini_options]\npythonpath = ["src"]\n')
    made.write(".gitignore", "__pycache__/\n.pytest_cache/\n")
    made.write("src/app.py", BASE_SOURCE)
    made.write("tests/test_app.py", BASE_TESTS)
    made.commit("base")
    return made


def commit_a_refusal(repo: Repo, tests: str) -> str:
    """Commit the source change that refuses "", along with `tests` for it."""
    repo.write("src/app.py", CHANGED_SOURCE)
    repo.append("tests/test_app.py", tests)
    return repo.commit("refuse an empty text")


# What the exit status says.


def test_a_test_that_fails_with_the_change_reversed_passes_the_check(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == [REFUSAL]


def test_what_each_test_failed_on_is_shown(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit)

    assert f"  {REFUSAL}\n    AssertionError: assert not True\n" in result.stdout


def test_a_test_that_passes_with_the_change_reversed_fails_the_check(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, UNNAMED) == [GUARD]
    assert listed_under(result, FAILS) == [REFUSAL]


def test_a_named_guard_may_pass_with_the_change_reversed(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST)

    result = repo.revert_check(commit, "--guard", GUARD)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, GUARDS) == [GUARD]


OTHER = "tests/test_app.py::test_other"
OTHER_TEST = '\n\ndef test_other():\n    assert accepts("y")\n'


def test_each_guard_takes_its_own_flag(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST + OTHER_TEST)

    result = repo.revert_check(commit, "--guard", GUARD, "--guard", OTHER)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, GUARDS) == [OTHER, GUARD]


def test_a_second_name_after_guard_is_a_named_test_and_no_guard(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST + OTHER_TEST)

    result = repo.revert_check(commit, "--guard", GUARD, OTHER, REFUSAL)

    assert result.returncode == 1
    assert listed_under(result, GUARDS) == [GUARD]
    assert listed_under(result, UNNAMED) == [OTHER]


def test_a_second_name_after_guard_cannot_drop_the_commits_other_tests(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST + OTHER_TEST)

    result = repo.revert_check(commit, "--guard", GUARD, REFUSAL)

    assert result.returncode == 1
    assert listed_under(result, LEFT_OUT) == [OTHER]


@pytest.mark.parametrize("everything", ["", ".", "./"])
def test_a_guard_that_covers_everything_is_refused(repo: Repo, everything: str) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST)

    result = repo.revert_check(commit, "--guard", everything)

    assert result.returncode == 2
    assert "--guard" in result.stderr


def test_an_empty_test_name_is_refused(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit, "")

    assert result.returncode == 2
    assert "empty" in result.stderr


def test_a_guard_name_does_not_cover_a_longer_name(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo,
        REFUSAL_TEST
        + """

def test_space():
    assert accepts(" ")


def test_space_too():
    assert accepts("  ")
""",
    )

    result = repo.revert_check(commit, "--guard", "tests/test_app.py::test_space")

    assert result.returncode == 1
    assert listed_under(result, UNNAMED) == ["tests/test_app.py::test_space_too"]


def test_every_parametrized_case_of_a_named_guard_counts(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo,
        """
import pytest


@pytest.mark.parametrize("text", ["a", "b"])
def test_still_accepts(text):
    assert accepts(text)
"""
        + REFUSAL_TEST,
    )

    result = repo.revert_check(commit, "--guard", "tests/test_app.py::test_still_accepts")

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, GUARDS) == [
        "tests/test_app.py::test_still_accepts[a]",
        "tests/test_app.py::test_still_accepts[b]",
    ]


def test_older_cases_beside_a_new_failing_case_fail_the_check_with_a_way_out(repo: Repo) -> None:
    repo.append(
        "tests/test_app.py",
        """

import pytest


@pytest.mark.parametrize(("text", "accepted"), [("a", True)])
def test_cases(text, accepted):
    assert accepts(text) is accepted
""",
    )
    repo.commit("add a test with one case")
    repo.write("src/app.py", CHANGED_SOURCE)
    path = repo.root / "tests" / "test_app.py"
    path.write_text(path.read_text().replace('[("a", True)]', '[("a", True), ("", False)]'))
    commit = repo.commit("refuse an empty text")
    older_case = "tests/test_app.py::test_cases[a-True]"

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, UNNAMED) == [older_case]
    assert "name those cases with --guard" in result.stdout

    named = repo.revert_check(commit, "--guard", older_case)

    assert named.returncode == 0, named.stdout + named.stderr
    assert "name those cases with --guard" not in named.stdout


def test_a_passing_test_with_no_failing_case_gets_no_such_way_out(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert "name those cases with --guard" not in result.stdout


def test_a_check_where_nothing_failed_fails(repo: Repo) -> None:
    commit = commit_a_refusal(repo, GUARD_TEST)

    result = repo.revert_check(commit, "--guard", GUARD)

    assert result.returncode == 1
    assert NOTHING_FAILED in result.stdout


# A test that did not run says nothing about the change.


def test_a_test_that_skips_with_the_change_reversed_did_not_run(repo: Repo) -> None:
    repo.write("src/app.py", BASE_SOURCE + "\n\nADDED = True\n")
    repo.append(
        "tests/test_app.py",
        """

import app
import pytest


def test_only_with_the_feature():
    if not hasattr(app, "ADDED"):
        pytest.skip("no feature")
    assert True
""",
    )
    commit = repo.commit("add a constant")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, DID_NOT_RUN) == ["tests/test_app.py::test_only_with_the_feature"]
    assert listed_under(result, FAILS) == []


def test_a_skip_beside_a_real_failure_fails_the_check(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE + "\n\nADDED = True\n")
    repo.append(
        "tests/test_app.py",
        REFUSAL_TEST
        + """

import app
import pytest


def test_only_with_the_feature():
    if not hasattr(app, "ADDED"):
        pytest.skip("no feature")
""",
    )
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, FAILS) == [REFUSAL]
    assert listed_under(result, DID_NOT_RUN) == ["tests/test_app.py::test_only_with_the_feature"]


@pytest.mark.parametrize(
    "never_runs",
    [
        'def test_never_runs():\n    if not hasattr(app, "ADDED"):\n        pytest.xfail("no")\n',
        (
            '@pytest.mark.xfail(not hasattr(app, "ADDED"), reason="no", run=False)\n'
            "def test_never_runs():\n    pass\n"
        ),
    ],
    ids=["called", "marked"],
)
def test_an_expected_failure_with_the_change_reversed_did_not_run(
    repo: Repo, never_runs: str
) -> None:
    repo.write("src/app.py", BASE_SOURCE + "\n\nADDED = True\n")
    repo.append("tests/test_app.py", f"\n\nimport app\nimport pytest\n\n\n{never_runs}")
    commit = repo.commit("add a constant")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, DID_NOT_RUN) == ["tests/test_app.py::test_never_runs"]
    assert listed_under(result, FAILS) == []


def test_a_test_that_kills_pytest_with_the_change_reversed_is_no_guard(repo: Repo) -> None:
    dies = "tests/test_app.py::test_dies"
    repo.write("src/app.py", CHANGED_SOURCE + "\n\nADDED = True\n")
    repo.append(
        "tests/test_app.py",
        REFUSAL_TEST
        + """

import os

import app


def test_dies():
    if not hasattr(app, "ADDED"):
        os._exit(0)
""",
    )
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit, "--guard", dies)

    assert result.returncode == 1
    assert listed_under(result, GUARDS) == []
    assert listed_under(result, DID_NOT_RUN) == [dies]


def test_pytest_crashing_with_the_change_reversed_fails_the_check(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE + "\n\nADDED = True\n")
    repo.write(
        "tests/conftest.py",
        """import app


def pytest_runtest_logfinish(nodeid, location):
    if not hasattr(app, "ADDED"):
        raise RuntimeError("the hook broke")
""",
    )
    repo.append("tests/test_app.py", REFUSAL_TEST)
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert len(listed_under(result, UNIMPORTED)) == 1


def test_a_case_that_disappears_with_the_change_reversed_did_not_run(repo: Repo) -> None:
    repo.write("src/names.py", 'NAMES = ["a", "b"]\n')
    repo.commit("add names")
    repo.write("src/names.py", 'NAMES = ["a", "b", "c"]\n')
    repo.write(
        "tests/test_names.py",
        """import pytest

from names import NAMES


@pytest.mark.parametrize("name", [n for n in NAMES if n != "a"])
def test_named(name):
    assert True
""",
    )
    commit = repo.commit("add a name")

    result = repo.revert_check(commit, "--guard", "tests/test_names.py::test_named")

    assert result.returncode == 1
    assert listed_under(result, DID_NOT_RUN) == ["tests/test_names.py::test_named[c]"]


def test_pytest_addopts_cannot_stop_a_run_early(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST)

    result = repo.revert_check(commit, PYTEST_ADDOPTS="-x")

    assert result.returncode == 1
    assert listed_under(result, UNNAMED) == [GUARD]


def test_a_test_file_the_reversal_stops_importing_fails_the_check(repo: Repo) -> None:
    repo.write("src/app.py", BASE_SOURCE + "\n\ndef added():\n    return 1\n")
    repo.write(
        "tests/test_added.py",
        "from app import added\n\n\ndef test_added():\n    assert added() == 1\n",
    )
    commit = repo.commit("add a function")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    (line,) = listed_under(result, UNIMPORTED)
    assert line.startswith("tests/test_added.py: ImportError: cannot import name 'added'")
    assert listed_under(result, FAILS) == []
    assert listed_under(result, DID_NOT_RUN) == []


def test_a_file_that_stops_importing_beside_a_real_failure_fails_the_check(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE + "\n\ndef added():\n    return 1\n")
    repo.write(
        "tests/test_added.py",
        "from app import added\n\n\ndef test_added():\n    assert added() == 1\n",
    )
    repo.append("tests/test_app.py", REFUSAL_TEST)
    commit = repo.commit("add a function")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, FAILS) == [REFUSAL]
    assert len(listed_under(result, UNIMPORTED)) == 1


NOTHING_BUT_THE_NAME = {
    "inside the test": """

def test_nothing():
    from app import LIMIT
    assert accepts("x") or LIMIT
""",
    "in a fixture": """

import pytest


@pytest.fixture
def limit():
    from app import LIMIT
    return LIMIT


def test_nothing(limit):
    assert True
""",
    "through the module": """

import app as module


def test_nothing():
    assert module.LIMIT
""",
    "patched by name": """

def test_nothing(monkeypatch):
    monkeypatch.setattr("app.LIMIT", 2)
""",
    "patched on the module": """

import app as module


def test_nothing(monkeypatch):
    monkeypatch.setattr(module, "LIMIT", 2)
""",
}


@pytest.mark.parametrize("test", NOTHING_BUT_THE_NAME.values(), ids=NOTHING_BUT_THE_NAME.keys())
def test_a_new_name_imported_late_shows_only_that_the_name_exists(repo: Repo, test: str) -> None:
    repo.write("src/app.py", CHANGED_SOURCE + "\n\nLIMIT = 1\n")
    repo.append("tests/test_app.py", test)
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, MISSING_NAME) == ["tests/test_app.py::test_nothing"]
    assert "'LIMIT'" in result.stdout
    assert listed_under(result, FAILS) == []
    assert NOTHING_FAILED not in result.stdout


def test_a_new_name_imported_late_beside_a_real_failure_fails_the_check(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE + "\n\nLIMIT = 1\n")
    repo.append("tests/test_app.py", REFUSAL_TEST + NOTHING_BUT_THE_NAME["inside the test"])
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, FAILS) == [REFUSAL]
    assert listed_under(result, MISSING_NAME) == ["tests/test_app.py::test_nothing"]


def test_an_attribute_error_on_a_value_is_a_real_failure(repo: Repo) -> None:
    repo.write("src/naming.py", "def name():\n    return None\n")
    repo.commit("add a name that is not there yet")
    repo.write("src/naming.py", 'def name():\n    return "x"\n')
    repo.write(
        "tests/test_naming.py",
        'from naming import name\n\n\ndef test_name():\n    assert name().upper() == "X"\n',
    )
    commit = repo.commit("give the name a value")

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == ["tests/test_naming.py::test_name"]


def test_a_new_keyword_reads_as_a_real_failure_with_what_it_failed_on(repo: Repo) -> None:
    repo.write("src/app.py", "def accepts(text, strict=False):\n    return True\n")
    repo.append("tests/test_app.py", '\n\ndef test_strict():\n    assert accepts("x", strict=True)\n')
    commit = repo.commit("add a keyword that does nothing")

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "unexpected keyword argument 'strict'" in result.stdout


def test_a_fix_for_a_missing_extra_reads_as_a_missing_name_too(repo: Repo) -> None:
    """The commit adds no name here, and the script cannot tell, so a person reads it."""
    repo.write("src/extra.py", "def accepts(text):\n    import not_installed_extra\n    return True\n")
    repo.commit("need an extra")
    repo.write(
        "src/extra.py",
        """def accepts(text):
    try:
        import not_installed_extra
    except ImportError:
        pass
    return True
""",
    )
    repo.write(
        "tests/test_extra.py",
        'from extra import accepts\n\n\ndef test_without_the_extra():\n    assert accepts("x")\n',
    )
    commit = repo.commit("work without the extra")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, MISSING_NAME) == ["tests/test_extra.py::test_without_the_extra"]


def test_a_new_module_imported_late_shows_only_that_the_module_exists(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE)
    repo.write("src/added.py", "x = 1\n")
    repo.append("tests/test_app.py", "\n\ndef test_nothing():\n    import added\n    assert added\n")
    commit = repo.commit("add a module")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, MISSING_NAME) == ["tests/test_app.py::test_nothing"]


def test_an_unrelated_new_import_cannot_hide_a_test_that_proves_nothing(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE + "\n\nLIMIT = 1\n")
    path = repo.root / "tests" / "test_app.py"
    path.write_text(
        path.read_text().replace("from app import accepts", "from app import LIMIT, accepts")
    )
    repo.append("tests/test_app.py", '\n\ndef test_nothing():\n    assert accepts("x") or LIMIT\n')
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 1


def test_a_conftest_the_reversal_stops_importing_fails_the_check(repo: Repo) -> None:
    repo.write("src/app.py", BASE_SOURCE + "\n\ndef added():\n    return 1\n")
    repo.write("tests/conftest.py", "from app import added\n")
    repo.append("tests/test_app.py", GUARD_TEST)
    commit = repo.commit("add a function and use it in a conftest")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, FAILS) == []


def test_one_file_failing_to_import_does_not_stop_another_files_tests(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE + "\n\ndef added():\n    return 1\n")
    repo.write(
        "tests/test_added.py",
        "from app import added\n\n\ndef test_added():\n    assert added() == 1\n",
    )
    repo.append("tests/test_app.py", GUARD_TEST)
    commit = repo.commit("add a function")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, UNNAMED) == [GUARD]


# With the change in place, a test has to pass for its failing to mean anything.


def test_a_test_that_already_fails_with_the_change_in_place_fails_the_check(repo: Repo) -> None:
    commit = commit_a_refusal(repo, '\n\ndef test_broken_either_way():\n    assert accepts("")\n')

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, NOT_PASSING) == ["tests/test_app.py::test_broken_either_way"]
    assert repo.worktrees() == 1


def test_a_test_that_skips_with_the_change_in_place_fails_the_check(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo, '\n\nimport pytest\n\n\ndef test_skipped():\n    pytest.skip("always")\n'
    )

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, NOT_PASSING) == ["tests/test_app.py::test_skipped"]


def test_a_test_that_kills_pytest_with_the_change_in_place_fails_the_check(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo, REFUSAL_TEST + "\n\nimport os\n\n\ndef test_dies():\n    os._exit(0)\n"
    )

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, NOT_PASSING) == ["tests/test_app.py::test_dies"]


def test_a_teardown_error_counts_against_the_test(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo,
        """

import pytest


@pytest.fixture
def breaks_afterwards():
    yield
    raise RuntimeError("teardown")


def test_with_a_bad_fixture(breaks_afterwards):
    assert not accepts("")
""",
    )

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, NOT_PASSING) == ["tests/test_app.py::test_with_a_bad_fixture"]


def test_a_test_id_holding_a_dash_is_read_whole(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo,
        """

import pytest


@pytest.mark.parametrize("broken", [True, False], ids=["a - broken", "a - fine"])
def test_dashes(broken):
    assert not broken and not accepts("")
""",
    )

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, NOT_PASSING) == ["tests/test_app.py::test_dashes[a - broken]"]


# Which tests run.


def test_by_default_only_tests_the_commit_added_or_changed_run(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == [REFUSAL]
    assert "test_older_one" not in result.stdout


def test_a_changed_test_runs_too_and_so_does_a_changed_decorator(repo: Repo) -> None:
    repo.append(
        "tests/test_app.py",
        """

import pytest


@pytest.mark.parametrize("text", ["a"])
def test_decorated(text):
    assert accepts(text)
""",
    )
    repo.commit("add a test to change later")
    repo.write("src/app.py", CHANGED_SOURCE)
    path = repo.root / "tests" / "test_app.py"
    path.write_text(
        path.read_text()
        .replace('["a"]', '["a", "b"]')
        .replace('assert accepts("x")', 'assert accepts("x") and not accepts("")')
    )
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit, "--guard", "tests/test_app.py::test_decorated")

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == [OLDER]
    assert "tests/test_app.py::test_decorated[b]" in listed_under(result, GUARDS)


def test_a_test_in_a_class_is_found(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo,
        """

class TestRefusal:
    def test_refuses_empty(self):
        assert not accepts("")
""",
    )

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == ["tests/test_app.py::TestRefusal::test_refuses_empty"]


def test_a_change_outside_any_test_function_is_pointed_out(repo: Repo) -> None:
    repo.append("tests/test_app.py", '\n\nCASES = ["a"]\n')
    repo.commit("add a list of cases")
    repo.write("src/app.py", CHANGED_SOURCE)
    path = repo.root / "tests" / "test_app.py"
    path.write_text(path.read_text().replace('["a"]', '["a", ""]') + REFUSAL_TEST)
    commit = repo.commit("refuse an empty text")
    line = path.read_text().splitlines().index('CASES = ["a", ""]') + 1

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, OUTSIDE) == [f"tests/test_app.py:{line}"]


def test_a_form_feed_earlier_in_the_file_does_not_hide_a_change(repo: Repo) -> None:
    """Python's splitlines breaks a line at a form feed, and git does not."""
    repo.write("src/app.py", CHANGED_SOURCE)
    path = repo.root / "tests" / "test_app.py"
    path.write_text("# one\x0ctwo\n\nCASES = 1\n" + path.read_text() + REFUSAL_TEST)
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert listed_under(result, OUTSIDE) == ["tests/test_app.py:3"]


def test_a_change_to_another_file_under_tests_is_pointed_out_and_not_run(repo: Repo) -> None:
    repo.write("tests/helpers.py", "def test_like():\n    return 1\n")
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, OUTSIDE) == ["tests/helpers.py"]
    assert listed_under(result, FAILS) == [REFUSAL]


def test_a_class_that_pytest_would_not_collect_is_pointed_out_and_not_run(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo, REFUSAL_TEST + "\n\nclass Helper:\n    def test_like(self):\n        return 1\n"
    )

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(listed_under(result, OUTSIDE)) == 1
    assert listed_under(result, FAILS) == [REFUSAL]


def test_a_class_header_or_docstring_is_not_pointed_out(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE)
    path = repo.root / "tests" / "test_app.py"
    path.write_text(
        '"""About these tests."""\n'
        + path.read_text()
        + '\n\nclass TestRefusal:\n    def test_refuses_empty(self):\n'
        + '        assert not accepts("")\n'
    )
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert OUTSIDE not in result.stdout


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            '@pytest.mark.parametrize("n", [1])\nclass TestCases:\n',
            '@pytest.mark.parametrize("n", [1, 2])\nclass TestCases:\n',
        ),
        (
            (
                "class TestCases:\n    @pytest.fixture(params=[1])\n    def n(self, request):\n"
                "        return request.param\n\n"
            ),
            (
                "class TestCases:\n    @pytest.fixture(params=[1, 2])\n    def n(self, request):\n"
                "        return request.param\n\n"
            ),
        ),
    ],
    ids=["on the class", "on its first fixture"],
)
def test_a_changed_decorator_in_a_test_class_is_pointed_out(
    repo: Repo, before: str, after: str
) -> None:
    body = "    def test_case(self, n):\n        assert n\n"
    repo.append("tests/test_app.py", f"\n\nimport pytest\n\n\n{before}{body}")
    repo.commit("add some cases")
    repo.write("src/app.py", CHANGED_SOURCE)
    path = repo.root / "tests" / "test_app.py"
    path.write_text(path.read_text().replace(before, after) + REFUSAL_TEST)
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(listed_under(result, OUTSIDE)) == 1


def test_a_new_import_or_blank_line_is_not_pointed_out(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE)
    path = repo.root / "tests" / "test_app.py"
    path.write_text(
        "import json\n# a comment\n    # an indented one\n" + path.read_text() + REFUSAL_TEST
    )
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert OUTSIDE not in result.stdout


def test_a_deleted_line_picks_no_test(repo: Repo) -> None:
    repo.append(
        "tests/test_app.py",
        '\n\ndef test_two_lines():\n    assert accepts("a")\n    assert accepts("b")\n',
    )
    repo.commit("add a test")
    repo.write("src/app.py", CHANGED_SOURCE)
    path = repo.root / "tests" / "test_app.py"
    path.write_text(path.read_text().replace('    assert accepts("b")\n', ""))
    commit = repo.commit("refuse an empty text and shorten a test")

    result = repo.revert_check(commit)

    assert result.returncode == 2
    assert "adds or changes no test" in result.stderr


def test_a_deleted_test_file_is_left_out(repo: Repo) -> None:
    repo.write("tests/test_old.py", "def test_old():\n    assert True\n")
    repo.commit("add a test file")
    (repo.root / "tests" / "test_old.py").unlink()
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == [REFUSAL]


def test_naming_some_of_a_commits_tests_checks_those_and_fails_on_the_rest(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + GUARD_TEST)

    result = repo.revert_check(commit, REFUSAL)

    assert result.returncode == 1
    assert listed_under(result, FAILS) == [REFUSAL]
    assert listed_under(result, LEFT_OUT) == [GUARD]


def test_naming_all_of_a_commits_tests_leaves_nothing_out(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST + REFUSAL_TEST.replace("empty", "nothing"))

    result = repo.revert_check(commit, REFUSAL, "tests/test_app.py::test_refuses_nothing")

    assert result.returncode == 0, result.stdout + result.stderr
    assert LEFT_OUT not in result.stdout


@pytest.mark.parametrize(
    "named",
    [
        ".",
        "./",
        "tests",
        "tests/",
        "tests/.",
        "tests/test_app.py",
        "./tests/test_app.py",
        "tests//test_app.py",
        "tests/../tests/test_app.py",
    ],
)
def test_a_named_file_or_directory_covers_its_tests(repo: Repo, named: str) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit, named)

    assert listed_under(result, NOT_PASSING) == []
    assert listed_under(result, FAILS) == [REFUSAL]
    assert listed_under(result, UNNAMED) == [OLDER]


def test_a_named_guard_runs_even_if_the_commit_did_not_touch_it(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit, "--guard", OLDER)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, GUARDS) == [OLDER]


# Where it runs.


def test_the_working_tree_is_left_as_it_was(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)
    repo.write("src/app.py", CHANGED_SOURCE + "# not committed\n")
    repo.write("scratch.txt", "untracked\n")
    before = repo.git("status", "--porcelain")

    assert repo.revert_check(commit).returncode == 0

    assert repo.git("status", "--porcelain") == before
    assert (repo.root / "src" / "app.py").read_text() == CHANGED_SOURCE + "# not committed\n"
    assert repo.worktrees() == 1


def test_a_commit_running_under_a_hook_keeps_what_is_staged(repo: Repo) -> None:
    """Git sets GIT_INDEX_FILE for a hook, and `git worktree add` would write to that index."""
    commit = commit_a_refusal(repo, REFUSAL_TEST)
    repo.write("scratch.txt", "staged\n")
    repo.git("add", "scratch.txt")

    result = repo.revert_check(commit, GIT_INDEX_FILE=str(repo.root / ".git" / "index"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert repo.git("diff", "--cached", "--name-only") == "scratch.txt"


def test_it_can_run_from_a_subdirectory(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit, cwd=repo.root / "tests")

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == [REFUSAL]


def test_a_later_commit_that_uses_the_reversed_lines_does_not_matter(repo: Repo) -> None:
    """Reversed at HEAD, the import in the later commit would fail every test in the file,
    and this test that proves nothing about the change would look as if it failed."""
    repo.write("src/app.py", BASE_SOURCE + '\n\ndef helper():\n    return ""\n')
    repo.append("tests/test_app.py", GUARD_TEST)
    commit = repo.commit("add a helper")
    repo.write("src/other.py", "from app import helper\n")
    path = repo.root / "tests" / "test_app.py"
    path.write_text("import other\n" + path.read_text())
    repo.commit("use the helper")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, UNNAMED) == [GUARD]
    assert listed_under(result, UNIMPORTED) == []


def test_at_runs_a_later_commits_tests_against_the_reversal(repo: Repo) -> None:
    commit = commit_a_refusal(repo, "")
    repo.append("tests/test_app.py", REFUSAL_TEST)
    later = repo.commit("test the refusal")

    result = repo.revert_check(commit, "--at", later, REFUSAL)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == [REFUSAL]


def test_at_may_leave_the_commits_own_tests_out(repo: Repo) -> None:
    commit = commit_a_refusal(repo, GUARD_TEST)
    repo.append("tests/test_app.py", REFUSAL_TEST)
    later = repo.commit("test the refusal")

    result = repo.revert_check(commit, "--at", later, REFUSAL)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, LEFT_OUT) == [GUARD]


def test_at_a_commit_the_reversal_does_not_apply_to_is_refused(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)
    repo.write("src/app.py", "def accepts(text):\n    return bool(text)\n")
    later = repo.commit("rewrite the same line")

    result = repo.revert_check(commit, "--at", later)

    assert result.returncode == 2
    assert f"{commit}'s source change does not reverse cleanly at {later}" in result.stderr
    assert repo.worktrees() == 1


def test_a_file_with_windows_line_endings_reverses(repo: Repo) -> None:
    repo.write("src/lines.py", "x = 1\r\ny = 2\r\n")
    repo.commit("add a file with CRLF")
    repo.write("src/lines.py", "x = 1\r\ny = 3\r\n")
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit, REFUSAL)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == [REFUSAL]


def test_a_source_file_that_is_not_utf8_reverses(repo: Repo) -> None:
    path = repo.root / "src" / "latin.py"
    path.write_bytes(b"# caf\xe9\nx = 1\n")
    repo.commit("add a file in latin-1")
    path.write_bytes(b"# caf\xe9\nx = 2\n")
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_change_of_the_same_size_is_not_hidden_by_old_bytecode(repo: Repo) -> None:
    """A pyc is valid for a source of the same size and second, so a baseline that left one
    could hide the reversal. Whether both land in one second is down to timing, so the nested
    test also says outright that no pyc gets written."""
    repo.write("src/answer.py", "def answer():\n    return 1\n")
    repo.commit("add an answer")
    repo.write("src/answer.py", "def answer():\n    return 2\n")
    repo.write(
        "tests/test_answer.py",
        """import sys

from answer import answer


def test_answer():
    assert sys.dont_write_bytecode
    assert answer() == 2
""",
    )
    commit = repo.commit("change the answer")

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == ["tests/test_answer.py::test_answer"]


@pytest.mark.parametrize(
    "left_behind", ["untracked.txt", "tracked.txt", "__pycache__/ignored.txt"]
)
def test_what_the_first_run_leaves_behind_is_gone_for_the_second(
    repo: Repo, left_behind: str
) -> None:
    """Otherwise a test that trips over its own file would look as if the reversal failed it."""
    repo.write("tracked.txt", "as committed\n")
    repo.commit("add a file")
    repo.write("src/app.py", BASE_SOURCE + "# a comment\n")
    repo.append(
        "tests/test_app.py",
        f"""

from pathlib import Path


def test_leaves_a_file():
    path = Path("{left_behind}")
    assert not path.exists() or path.read_text() == "as committed\\n"
    path.parent.mkdir(exist_ok=True)
    path.write_text("left behind")
""",
    )
    commit = repo.commit("a change that does nothing")

    result = repo.revert_check(commit)

    assert result.returncode == 1
    assert listed_under(result, UNNAMED) == ["tests/test_app.py::test_leaves_a_file"]
    assert listed_under(result, FAILS) == []


def test_a_failure_message_the_terminal_cannot_print_is_escaped(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo,
        """

import pytest


def test_refuses_empty():
    if accepts(""):
        pytest.fail("bad \\ud83d text")
""",
    )

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "bad \\ud83d text" in result.stdout


def test_a_binary_file_under_src_reverses(repo: Repo) -> None:
    (repo.root / "src" / "data.bin").write_bytes(bytes(range(256)))
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("stop", [signal.SIGTERM, signal.SIGHUP], ids=["term", "hangup"])
def test_being_told_to_stop_removes_the_temporary_checkout(
    repo: Repo, tmp_path: Path, stop: signal.Signals
) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)
    hanging = tmp_path / "hanging"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), commit],
        cwd=repo.root,
        env={**repo.env, "UV_HANGS": str(hanging), "TMPDIR": str(scratch)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while not hanging.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert hanging.exists()

    process.send_signal(stop)

    assert process.wait(timeout=20) != 0
    assert repo.worktrees() == 1
    assert list(scratch.iterdir()) == []


# When the check cannot run.


def test_a_name_that_is_not_a_commit_is_refused(repo: Repo) -> None:
    result = repo.revert_check("no-such-commit")

    assert result.returncode == 2
    assert "no-such-commit is not a commit" in result.stderr


def test_a_range_is_not_a_commit(repo: Repo) -> None:
    commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check("HEAD~1..HEAD")

    assert result.returncode == 2
    assert "is not a commit" in result.stderr


def test_a_commit_with_nothing_to_reverse_is_refused(repo: Repo) -> None:
    repo.append("tests/test_app.py", GUARD_TEST)
    commit = repo.commit("only a test")

    result = repo.revert_check(commit)

    assert result.returncode == 2
    assert "changes nothing under src" in result.stderr


def test_a_merge_commit_is_refused(repo: Repo) -> None:
    repo.git("switch", "-q", "-c", "side")
    repo.write("src/side.py", "x = 1\n")
    repo.commit("side")
    repo.git("switch", "-q", "main")
    repo.write("src/main.py", "x = 2\n")
    repo.commit("main")
    repo.git("merge", "-q", "--no-edit", "side")

    result = repo.revert_check("HEAD")

    assert result.returncode == 2
    assert "is a merge" in result.stderr


def test_a_commit_that_touches_no_test_needs_them_named(repo: Repo) -> None:
    repo.write("src/app.py", CHANGED_SOURCE)
    commit = repo.commit("no test in this one")

    result = repo.revert_check(commit)

    assert result.returncode == 2
    assert "adds or changes no test" in result.stderr


def test_a_fixture_named_like_a_test_is_pointed_out_and_not_run(repo: Repo) -> None:
    commit = commit_a_refusal(
        repo,
        REFUSAL_TEST + "\n\nimport pytest\n\n\n@pytest.fixture\ndef test_data():\n    return 1\n",
    )

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == [REFUSAL]
    assert len(listed_under(result, OUTSIDE)) == 1


def test_a_named_test_that_does_not_exist_is_refused(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit, "tests/test_app.py::test_no_such_thing")

    assert result.returncode == 2
    assert "test_no_such_thing" in result.stderr
    assert repo.worktrees() == 1


@pytest.mark.parametrize(
    "broken",
    [b"def test_broken(:\n", b'x = "caf\xe9"\ndef test_broken():\n    pass\n'],
    ids=["syntax", "encoding"],
)
def test_a_test_file_that_does_not_parse_is_refused(repo: Repo, broken: bytes) -> None:
    repo.write("src/app.py", CHANGED_SOURCE)
    (repo.root / "tests" / "test_broken.py").write_bytes(broken)
    commit = repo.commit("a test file python cannot read")

    result = repo.revert_check(commit)

    assert result.returncode == 2
    assert "tests/test_broken.py" in result.stderr


@pytest.mark.parametrize(
    "start", [b"# -*- coding: latin-1 -*-\n# caf\xe9\n", b"\xef\xbb\xbf"], ids=["cookie", "bom"]
)
def test_a_test_file_that_declares_its_encoding_is_read(repo: Repo, start: bytes) -> None:
    repo.write("src/app.py", CHANGED_SOURCE)
    (repo.root / "tests" / "test_declared.py").write_bytes(
        start + b'from app import accepts\n\n\ndef test_refuses():\n    assert not accepts("")\n'
    )
    commit = repo.commit("refuse an empty text")

    result = repo.revert_check(commit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert listed_under(result, FAILS) == ["tests/test_declared.py::test_refuses"]


def test_uv_failing_before_pytest_starts_is_refused_with_its_reason(repo: Repo) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit, UV_BROKEN="No solution found")

    assert result.returncode == 2
    assert "No solution found" in result.stderr
    assert repo.worktrees() == 1


def test_a_crash_says_the_check_could_not_run(repo: Repo, tmp_path: Path) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)
    no_git = tmp_path / "no-git"
    no_git.mkdir()

    result = repo.revert_check(commit, PATH=str(no_git))

    assert result.returncode == 2
    assert "could not finish" in result.stderr


def test_a_report_it_cannot_read_says_the_check_could_not_run(repo: Repo) -> None:
    repo.write(
        "tests/conftest.py",
        """import os


def pytest_sessionstart(session):
    with open(os.environ["REVERT_CHECK_LOG"], "a") as log:
        log.write("not json\\n")
""",
    )
    commit = commit_a_refusal(repo, REFUSAL_TEST)

    result = repo.revert_check(commit)

    assert result.returncode == 2
    assert "could not finish" in result.stderr
    assert repo.worktrees() == 1


def test_no_uv_on_the_path_is_refused(repo: Repo, tmp_path: Path) -> None:
    commit = commit_a_refusal(repo, REFUSAL_TEST)
    only_git = tmp_path / "only-git"
    only_git.mkdir()
    git = shutil.which("git")
    assert git is not None
    (only_git / "git").symlink_to(git)

    result = repo.revert_check(commit, PATH=str(only_git))

    assert result.returncode == 2
    assert "could not run uv" in result.stderr
    assert repo.worktrees() == 1
