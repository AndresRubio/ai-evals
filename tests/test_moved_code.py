"""Tests for scripts/moved_code.py, run against a small temporary repo.

The base is a tree written from the index, which the script reads as it reads a commit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "moved_code.py"

RATES = '''"""Rates and their intervals."""

import math
from dataclasses import dataclass

# Below this share of rows the rate is not reported.
MIN_SHARE = 0.5


@dataclass(frozen=True)
class Rate:
    value: float


def rate(passed: int, total: int) -> Rate:
    """The share that passed."""
    # An empty batch has no rate.
    if total == 0:
        raise ValueError("no rows")
    return Rate(passed / total)


def _interval(value: float, total: int) -> float:
    return 1.96 * math.sqrt(value * (1 - value) / total)
'''

# RATES split in two, every statement as it was.
HALF_RATE = '''"""Rates and their intervals."""

from dataclasses import dataclass

# Below this share of rows the rate is not reported.
MIN_SHARE = 0.5


@dataclass(frozen=True)
class Rate:
    value: float


def rate(passed: int, total: int) -> Rate:
    """The share that passed."""
    # An empty batch has no rate.
    if total == 0:
        raise ValueError("no rows")
    return Rate(passed / total)
'''

HALF_INTERVAL = """import math


def _interval(value: float, total: int) -> float:
    return 1.96 * math.sqrt(value * (1 - value) / total)
"""


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q")

    def git(self, *args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    def write(self, path: str, text: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def base(self) -> str:
        """The tree of everything written so far."""
        self.git("add", "-A")
        return self.git("write-tree")

    def moved(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path)


def split(repo: Repo, rate_half: str = HALF_RATE, interval_half: str = HALF_INTERVAL) -> str:
    """rates.py at the base, and split into two modules of a package now."""
    repo.write("src/pkg/rates.py", RATES)
    base = repo.base()
    (repo.root / "src/pkg/rates.py").unlink()
    repo.write("src/pkg/stats/rate.py", rate_half)
    repo.write("src/pkg/stats/interval.py", interval_half)
    return base


def test_a_module_split_in_two_with_every_statement_as_it_was_differs_in_nothing(
    repo: Repo,
) -> None:
    result = repo.moved(split(repo))

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "5 names compared, 0 differ\n"


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [
        ("if total == 0:", "if total <= 0:", "rate"),
        ("# An empty batch has no rate.", "# No rows, no rate.", "rate"),
        ('"""The share that passed."""', '"""The pass rate."""', "rate"),
        ("# Below this share of rows the rate is not reported.", "# The floor.", "MIN_SHARE"),
        ("@dataclass(frozen=True)", "@dataclass", "Rate"),
        ('"""Rates and their intervals."""', '"""Rates."""', "<module docstring>"),
    ],
)
def test_a_changed_line_is_reported_under_the_name_that_holds_it(
    repo: Repo, old: str, new: str, name: str
) -> None:
    """A condition, a comment in a body, a docstring, a comment above a constant, a
    decorator and the module's own docstring."""
    assert old in HALF_RATE

    result = repo.moved(split(repo, rate_half=HALF_RATE.replace(old, new)))

    assert result.returncode == 1
    assert result.stdout == (
        f"changed {name}  src/pkg/rates.py -> src/pkg/stats/rate.py\n5 names compared, 1 differ\n"
    )


def test_the_differing_lines_are_printed_on_request(repo: Repo) -> None:
    base = split(repo, rate_half=HALF_RATE.replace("if total == 0:", "if total <= 0:"))

    result = repo.moved(base, "--diff")

    assert result.returncode == 1
    assert "-    if total == 0:\n+    if total <= 0:\n" in result.stdout


def test_a_statement_that_did_not_arrive_is_gone_and_one_nobody_had_is_new(repo: Repo) -> None:
    extra = "\n\ndef _half_width(value: float) -> float:\n    return value / 2\n"
    interval = "import math\n" + extra

    result = repo.moved(split(repo, interval_half=interval))

    assert result.returncode == 1
    assert result.stdout == (
        "new     _half_width  src/pkg/stats/interval.py\n"
        "gone    _interval  src/pkg/rates.py\n"
        "6 names compared, 2 differ\n"
    )


def test_a_rename_named_on_the_command_line_compares_equal_and_is_otherwise_new_and_gone(
    repo: Repo,
) -> None:
    base = split(repo, interval_half=HALF_INTERVAL.replace("_interval", "interval"))

    unnamed = repo.moved(base)
    named = repo.moved(base, "--rename", "_interval=interval")

    assert unnamed.returncode == 1
    assert "gone    _interval  src/pkg/rates.py\n" in unnamed.stdout
    assert "new     interval  src/pkg/stats/interval.py\n" in unnamed.stdout
    assert named.returncode == 0, named.stdout
    assert named.stdout == (
        "--rename _interval=interval: 1 use renamed at the base\n5 names compared, 0 differ\n"
    )


def test_a_rename_leaves_the_rest_of_the_renamed_statement_checked(repo: Repo) -> None:
    interval = HALF_INTERVAL.replace("_interval", "interval").replace("1.96", "2.58")

    result = repo.moved(split(repo, interval_half=interval), "--rename", "_interval=interval")

    assert result.returncode == 1
    assert "changed interval  src/pkg/rates.py -> src/pkg/stats/interval.py\n" in result.stdout


def test_a_rename_reaches_code_alone_and_only_that_exact_name(repo: Repo) -> None:
    """`rate` is renamed where code calls it. `Rate`, the comments that say "rate" and a
    string that holds the word stay as they were, so a file that kept them compares equal."""
    more = '\n\nLABEL = "rate"\n\n\ndef both(passed: int) -> Rate:\n    return rate(passed, 2)\n'
    repo.write("src/pkg/rates.py", RATES + more)
    base = repo.base()
    now = (
        (RATES + more)
        .replace("def rate(", "def pass_rate(")
        .replace("return rate(", "return pass_rate(")
    )
    repo.write("src/pkg/rates.py", now)

    result = repo.moved(base, "--rename", "rate=pass_rate")

    assert result.returncode == 0, result.stdout
    assert result.stdout == (
        "--rename rate=pass_rate: 2 uses renamed at the base\n7 names compared, 0 differ\n"
    )


def test_a_comment_reworded_for_a_rename_still_shows(repo: Repo) -> None:
    repo.write("src/pkg/rates.py", RATES)
    base = repo.base()
    now = RATES.replace("def rate(", "def pass_rate(").replace("has no rate.", "has no pass rate.")
    repo.write("src/pkg/rates.py", now)

    result = repo.moved(base, "--rename", "rate=pass_rate")

    assert result.returncode == 1
    assert result.stdout == (
        "changed pass_rate  src/pkg/rates.py\n"
        "--rename rate=pass_rate: 1 use renamed at the base\n"
        "5 names compared, 1 differ\n"
    )


def test_a_rename_leaves_an_attribute_and_a_keyword_of_the_same_name_alone(repo: Repo) -> None:
    """The commit renames the function `run`. It also renames a method call and a keyword
    that only share the name, and the rename on the command line does not hide those."""
    code = (
        "def run(job: object) -> None:\n    job.run()\n\n\n"
        "def start(job: object) -> None:\n    launch(job, run=True)\n    run(job)\n"
    )
    repo.write("src/pkg/a.py", code)
    base = repo.base()
    repo.write("src/pkg/a.py", code.replace("run", "execute"))

    result = repo.moved(base, "--rename", "run=execute")

    assert result.returncode == 1
    assert result.stdout == (
        "changed execute  src/pkg/a.py\n"
        "changed start  src/pkg/a.py\n"
        "--rename run=execute: 2 uses renamed at the base\n"
        "2 names compared, 2 differ\n"
    )


def test_two_modules_that_swap_a_constant_both_changed(repo: Repo) -> None:
    repo.write("src/pkg/a.py", "LIMIT = 5\n")
    repo.write("src/pkg/b.py", "LIMIT = 50\n")
    base = repo.base()
    repo.write("src/pkg/a.py", "LIMIT = 50\n")
    repo.write("src/pkg/b.py", "LIMIT = 5\n")

    result = repo.moved(base)

    assert result.returncode == 1
    assert result.stdout == (
        "changed LIMIT  src/pkg/a.py\nchanged LIMIT  src/pkg/b.py\n1 names compared, 2 differ\n"
    )


CHECK = "def check(n: int) -> bool:\n    return n < LIMIT\n"


def test_a_function_moved_beside_another_statement_of_a_name_it_uses_is_listed(repo: Repo) -> None:
    """`check` reads LIMIT. Its text is the same in b.py, where LIMIT is 50."""
    repo.write("src/pkg/a.py", "LIMIT = 5\n\n\n" + CHECK)
    repo.write("src/pkg/b.py", "LIMIT = 50\n")
    base = repo.base()
    repo.write("src/pkg/a.py", "LIMIT = 5\n")
    repo.write("src/pkg/b.py", "LIMIT = 50\n\n\n" + CHECK)

    result = repo.moved(base)

    assert result.returncode == 1
    assert result.stdout == (
        "moved   check  src/pkg/a.py -> src/pkg/b.py  (LIMIT is another statement there)\n"
        "2 names compared, 1 differ\n"
    )


def test_a_function_moved_with_the_statement_it_uses_is_not_listed(repo: Repo) -> None:
    repo.write("src/pkg/a.py", "LIMIT = 5\n\n\n" + CHECK)
    base = repo.base()
    (repo.root / "src/pkg/a.py").unlink()
    repo.write("src/pkg/b.py", "LIMIT = 5\n\n\n" + CHECK)

    result = repo.moved(base)

    assert result.returncode == 0, result.stdout
    assert result.stdout == "2 names compared, 0 differ\n"


def test_a_moved_statement_that_reads_its_modules_name_or_file_is_listed(repo: Repo) -> None:
    """The logger's name and the folder change with the module. An attribute of that
    name, as on an exception's type, does not."""
    code = (
        "import logging\nfrom pathlib import Path\n\n"
        "log = logging.getLogger(__name__)\n"
        "STATIC = Path(__file__).parent / 'static'\n\n\n"
        "def kind(error: Exception) -> str:\n    return type(error).__name__\n"
    )
    repo.write("src/pkg/a.py", code)
    base = repo.base()
    (repo.root / "src/pkg/a.py").unlink()
    repo.write("src/pkg/sub/a.py", code)

    result = repo.moved(base)

    assert result.returncode == 1
    assert result.stdout == (
        "moved   STATIC  src/pkg/a.py -> src/pkg/sub/a.py  (reads __file__)\n"
        "moved   log  src/pkg/a.py -> src/pkg/sub/a.py  (reads __name__)\n"
        "3 names compared, 2 differ\n"
    )


@pytest.mark.parametrize(
    ("moved", "beside", "note"),
    [
        (
            "def size(xs: list[int]) -> int:\n    return len(xs)\n",
            "def len(xs: list[int]) -> int:\n    return 0\n",
            "len is a statement there",
        ),
        (
            "def load() -> object:\n    from .store import S\n\n    return S\n",
            "",
            "imports from .store",
        ),
        ("def where() -> str | None:\n    return __package__\n", "", "reads __package__"),
        (
            "def bump() -> None:\n    global COUNT\n    COUNT += 1\n",
            "",
            "declares global COUNT",
        ),
    ],
)
def test_a_moved_statement_its_new_module_may_change_is_listed(
    repo: Repo, moved: str, beside: str, note: str
) -> None:
    """A builtin the new module shadows, a relative import, the module's package and a
    global: each keeps its text and may do something else in another module."""
    name = moved.split("(")[0].removeprefix("def ")
    repo.write("src/pkg/a.py", "COUNT = 0\n\n\n" + moved)
    repo.write("src/pkg/sub/b.py", "COUNT = 0\n\n\n" + beside)
    base = repo.base()
    repo.write("src/pkg/a.py", "COUNT = 0\n")
    repo.write("src/pkg/sub/b.py", "COUNT = 0\n\n\n" + beside + "\n\n" + moved)

    result = repo.moved(base)

    assert result.returncode == 1
    assert f"moved   {name}  src/pkg/a.py -> src/pkg/sub/b.py  ({note})\n" in result.stdout
    assert result.stdout.count("\n") == 2


def test_a_stub_file_is_compared(repo: Repo) -> None:
    repo.write("src/pkg/a.pyi", "def f(x: int) -> int: ...\n")
    base = repo.base()
    repo.write("src/pkg/a.pyi", "def f(x: str) -> int: ...\n")

    result = repo.moved(base)

    assert result.returncode == 1
    assert result.stdout == "changed f  src/pkg/a.pyi\n1 names compared, 1 differ\n"


def test_a_comment_on_an_import_line_is_compared(repo: Repo) -> None:
    repo.write("src/pkg/a.py", "import os  # kept for the plugin loader\n\nX = os.sep\n")
    base = repo.base()
    repo.write("src/pkg/a.py", "import os\n\nX = os.sep\n")

    result = repo.moved(base)

    assert result.returncode == 1
    assert result.stdout == (
        "gone    <comment at an import>  src/pkg/a.py\n2 names compared, 1 differ\n"
    )


def test_a_file_that_cannot_be_read_as_text_exits_2(repo: Repo) -> None:
    repo.write("src/pkg/a.py", "X = 1\n")
    base = repo.base()
    (repo.root / "src/pkg/b.py").write_bytes(b"X = '\xff\xfe'\n")

    result = repo.moved(base)

    assert result.returncode == 2
    assert "UnicodeDecodeError" in result.stderr


def test_an_import_at_the_top_is_left_out_and_one_inside_a_function_is_not(repo: Repo) -> None:
    lazy = "def load() -> object:\n    from pkg.old import thing\n\n    return thing\n"
    repo.write("src/pkg/a.py", "import os\nfrom pkg.old import other\n\n" + lazy)
    base = repo.base()
    repo.write("src/pkg/a.py", "import sys\nfrom pkg.new import other\n\n" + lazy)

    top_only = repo.moved(base)
    repo.write("src/pkg/a.py", "import os\n\n" + lazy.replace("pkg.old", "pkg.new"))
    inside = repo.moved(base)

    assert top_only.returncode == 0, top_only.stdout
    assert inside.returncode == 1
    assert inside.stdout == "changed load  src/pkg/a.py\n1 names compared, 1 differ\n"


def test_a_name_two_modules_define_is_matched_text_by_text(repo: Repo) -> None:
    one = "def load() -> int:\n    return 1\n"
    two = "def load() -> int:\n    return 2\n"
    repo.write("src/pkg/a.py", one)
    repo.write("src/pkg/b.py", two)
    base = repo.base()
    (repo.root / "src/pkg/a.py").unlink()
    (repo.root / "src/pkg/b.py").unlink()
    repo.write("src/pkg/x.py", two)
    repo.write("src/pkg/y.py", one.replace("return 1", "return 10"))

    result = repo.moved(base)

    assert result.returncode == 1
    assert (
        result.stdout == "changed load  src/pkg/a.py -> src/pkg/y.py\n1 names compared, 1 differ\n"
    )


def test_a_statement_that_defines_no_name_goes_by_its_first_line(repo: Repo) -> None:
    guard = 'if __name__ == "__main__":\n    raise SystemExit(0)\n'
    repo.write("src/pkg/a.py", guard)
    base = repo.base()
    repo.write("src/pkg/a.py", guard.replace("SystemExit(0)", "SystemExit(1)"))

    result = repo.moved(base)

    assert result.returncode == 1
    assert result.stdout.startswith('changed <if __name__ == "__main__":>  src/pkg/a.py\n')


def test_a_comment_that_belongs_to_no_statement_is_compared_too(repo: Repo) -> None:
    repo.write("src/pkg/a.py", "# Why this import is here.\nimport os\n\nX = 1\n# The end.\n")
    base = repo.base()
    repo.write("src/pkg/a.py", "import os\n\nX = 1\n")

    result = repo.moved(base)

    assert result.returncode == 1
    assert result.stdout == (
        "gone    <comment at an import>  src/pkg/a.py\n"
        "gone    <comment at the end>  src/pkg/a.py\n"
        "3 names compared, 2 differ\n"
    )


def test_only_the_named_paths_are_read_and_src_is_the_default(repo: Repo) -> None:
    repo.write("src/pkg/a.py", "X = 1\n")
    repo.write("tests/test_a.py", "def test_x() -> None:\n    assert True\n")
    base = repo.base()
    repo.write("tests/test_a.py", "def test_x() -> None:\n    assert False\n")
    repo.write("notes/draft.py", "this is not python")

    assert repo.moved(base).stdout == "1 names compared, 0 differ\n"
    with_tests = repo.moved(base, "src", "tests")
    assert with_tests.returncode == 1
    assert "changed test_x  tests/test_a.py\n" in with_tests.stdout


def test_a_file_git_ignores_is_not_read(repo: Repo) -> None:
    repo.write(".gitignore", "build/\n")
    repo.write("src/pkg/a.py", "X = 1\n")
    base = repo.base()
    repo.write("src/pkg/build/copy.py", "X = 2\n")

    assert repo.moved(base).stdout == "1 names compared, 0 differ\n"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["no-such-revision"], "moved_code: "),
        (["BASE", "--rename", "percent"], "--rename takes old=new"),
    ],
)
def test_a_run_that_cannot_compare_exits_2(repo: Repo, args: list[str], message: str) -> None:
    repo.write("src/pkg/a.py", "X = 1\n")
    base = repo.base()

    result = repo.moved(*[base if arg == "BASE" else arg for arg in args])

    assert result.returncode == 2
    assert message in result.stderr
    assert result.stdout == ""


def test_a_file_that_does_not_parse_exits_2_and_is_named(repo: Repo) -> None:
    repo.write("src/pkg/a.py", "X = 1\n")
    base = repo.base()
    repo.write("src/pkg/a.py", "def broken(:\n")

    result = repo.moved(base)

    assert result.returncode == 2
    assert "src/pkg/a.py does not parse" in result.stderr


def test_a_run_from_a_folder_below_the_top_exits_2(repo: Repo) -> None:
    repo.write("src/pkg/a.py", "X = 1\n")
    base = repo.base()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), base],
        cwd=repo.root / "src",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "run this from the top of the repository" in result.stderr
