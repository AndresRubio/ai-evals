"""Tests for scripts/weaken_rules.py, run against a small temporary project.

The project's own tests call the real lint-imports, as tests/test_layers.py does, so a
change to its contract is caught or survives for the reason it would here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "weaken_rules.py"

PYPROJECT = """[project]
name = "pkg"

[tool.importlinter]
root_package = "pkg"

# high may import low, and low knows nothing of high.
[[tool.importlinter.contracts]]
name = "pkg is layered"
type = "layers"
containers = ["pkg"]
exhaustive = true
layers = [
    "high",
    "low",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

# The project's tests: the package as it stands, and one import the order refuses.
PROJECT_TESTS = '''import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINT_IMPORTS = Path(sys.executable).with_name("lint-imports")


def broken(tmp_path, module=None, text=""):
    """Whether lint-imports refuses a copy of the package, with one module written over."""
    shutil.copytree(REPO / "src", tmp_path / "src")
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    if module:
        (tmp_path / "src" / "pkg" / module).write_text(text)
    done = subprocess.run(
        [str(LINT_IMPORTS), "--no-cache", "--config", "pyproject.toml"],
        cwd=tmp_path,
        env={"PATH": str(LINT_IMPORTS.parent), "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
    )
    assert "Contracts:" in done.stdout, done.stdout + done.stderr
    return done.returncode != 0


def test_the_package_as_it_stands_is_kept(tmp_path):
    assert not broken(tmp_path)


def test_low_importing_high_is_refused(tmp_path):
    assert broken(tmp_path, "low.py", "from pkg import high\\n")
'''

EXHAUSTIVE_TEST = """

def test_a_module_with_no_row_is_refused(tmp_path):
    assert broken(tmp_path, "extra.py")
"""


class Project:
    def __init__(self, root: Path, pyproject: str = PYPROJECT) -> None:
        self.root = root
        self.write("pyproject.toml", pyproject)
        self.write("src/pkg/__init__.py", "")
        self.write("src/pkg/high.py", "from pkg import low\n")
        self.write("src/pkg/low.py", "")

    def write(self, path: str, text: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def weaken(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--workers", "0", *args],
            cwd=cwd or self.root,
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project(tmp_path)


def test_tests_that_pin_every_rule_catch_every_change(project: Project) -> None:
    project.write("tests/test_rules.py", PROJECT_TESTS + EXHAUSTIVE_TEST)

    result = project.weaken("tests/test_rules.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == (
        "caught    pkg is layered: no import refused\n"
        "caught    pkg is layered: no longer exhaustive\n"
        "caught    pkg is layered: 'high' and 'low' swapped\n"
        "caught    pkg is layered: 'high' and 'low' may import each other\n"
        "4 changes tried, 0 survived\n"
    )


def test_a_rule_no_test_pins_survives_and_fails_the_run(project: Project) -> None:
    project.write("tests/test_rules.py", PROJECT_TESTS)

    result = project.weaken("tests/test_rules.py")

    assert result.returncode == 1
    assert result.stdout == (
        "caught    pkg is layered: no import refused\n"
        "SURVIVED  pkg is layered: no longer exhaustive\n"
        "caught    pkg is layered: 'high' and 'low' swapped\n"
        "caught    pkg is layered: 'high' and 'low' may import each other\n"
        "4 changes tried, 1 survived\n"
    )


def test_the_project_is_left_as_it_was(project: Project) -> None:
    project.write("tests/test_rules.py", PROJECT_TESTS)
    before = {path: path.read_bytes() for path in project.root.rglob("*") if path.is_file()}

    project.weaken("tests/test_rules.py", "--only", "swapped")

    after = {path: path.read_bytes() for path in project.root.rglob("*") if path.is_file()}
    assert after == before


def test_only_keeps_the_changes_whose_line_holds_the_text(project: Project) -> None:
    project.write("tests/test_rules.py", PROJECT_TESTS)

    result = project.weaken("tests/test_rules.py", "--only", "exhaustive")

    assert result.returncode == 1
    assert result.stdout == (
        "SURVIVED  pkg is layered: no longer exhaustive\n1 changes tried, 1 survived\n"
    )


def test_tests_that_fail_on_the_rules_as_they_stand_prove_nothing(project: Project) -> None:
    project.write(
        "tests/test_rules.py", PROJECT_TESTS + "\n\ndef test_broken():\n    assert False\n"
    )

    result = project.weaken("tests/test_rules.py")

    assert result.returncode == 2
    assert "the tests fail on the rules as they stand: 1 failed, 2 passed" in result.stderr
    assert result.stdout == ""


def test_a_test_file_that_collects_nothing_proves_nothing(project: Project) -> None:
    project.write("tests/test_rules.py", "")

    result = project.weaken("tests/test_rules.py")

    assert result.returncode == 2
    assert "pytest exited 5" in result.stderr


RICHER = """[tool.importlinter]
root_package = "pkg"

[[tool.importlinter.contracts]]
name = "layers"
type = "layers"
containers = ["pkg"]
layers = ["top", "a | b", "base"]

[[tool.importlinter.contracts]]
name = "three share a row"
type = "layers"
containers = ["pkg.sub"]
exhaustive = true
layers = ["x | y | z"]

[[tool.importlinter.contracts]]
name = "walls"
type = "forbidden"
source_modules = ["pkg.a", "pkg.b"]
forbidden_modules = ["pkg.web"]
ignore_imports = ["pkg.a -> pkg.web.types", "pkg.b -> pkg.web.types"]

[[tool.importlinter.contracts]]
name = "apart"
type = "independence"
modules = ["pkg.a", "pkg.b"]
"""


def test_the_changes_listed_for_each_kind_of_contract(tmp_path: Path) -> None:
    """A row shared by two modules gives each order once, and is only swapped with its
    neighbour, since joining the two rows would also join the shared row's modules. A list of one module is left
    alone, since dropping it would empty the rule. A kind of contract the script does
    not know is removed, and the two kinds it knows stay in the file with nothing left
    to refuse."""
    project = Project(tmp_path, RICHER)

    result = project.weaken("--list")

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "layers: no import refused",
        "layers: 'top' and 'a | b' swapped",
        "layers: 'a | b' and 'base' swapped",
        "layers: the modules of 'a | b' may import each other",
        "layers: 'a' above 'b'",
        "layers: 'a' below 'b'",
        "three share a row: no import refused",
        "three share a row: no longer exhaustive",
        "three share a row: the modules of 'x | y | z' may import each other",
        "three share a row: 'x' above 'y | z'",
        "three share a row: 'x' below 'y | z'",
        "three share a row: 'y' above 'x | z'",
        "three share a row: 'y' below 'x | z'",
        "three share a row: 'z' above 'x | y'",
        "three share a row: 'z' below 'x | y'",
        "walls: no import refused",
        "walls: pkg.a no longer a source",
        "walls: pkg.b no longer a source",
        "walls: 'pkg.a -> pkg.web.types' no longer let through",
        "walls: 'pkg.b -> pkg.web.types' no longer let through",
        "apart: removed",
    ]


def test_two_plain_neighbours_may_also_import_each_other(tmp_path: Path) -> None:
    project = Project(tmp_path, RICHER.replace('"top", "a | b", "base"', '"top", "mid", "base"'))

    listed = project.weaken("--list", "--only", "layers: ").stdout.splitlines()

    assert listed == [
        "layers: no import refused",
        "layers: 'top' and 'mid' swapped",
        "layers: 'top' and 'mid' may import each other",
        "layers: 'mid' and 'base' swapped",
        "layers: 'mid' and 'base' may import each other",
    ]


@pytest.mark.parametrize(
    ("pyproject", "args", "message"),
    [
        (PYPROJECT, ["tests/test_missing.py"], "no such test file: tests/test_missing.py"),
        (PYPROJECT, ["--only", "no such change"], "no change to try"),
        ('[tool.importlinter]\nroot_package = "pkg"\n', [], "no change to try"),
        (
            PYPROJECT.replace("exhaustive = true", "exhaustive = true\nweight = 1.5"),
            ["--list"],
            "a value this script cannot write back: 1.5",
        ),
        (
            PYPROJECT.replace('name = "pkg is layered"', 'name = "pkg is layered'),
            ["--list"],
            "pyproject.toml does not parse",
        ),
    ],
)
def test_a_run_that_would_prove_nothing_exits_2(
    tmp_path: Path, pyproject: str, args: list[str], message: str
) -> None:
    project = Project(tmp_path, pyproject)

    result = project.weaken(*args)

    assert result.returncode == 2
    assert message in result.stderr
    assert result.stdout == ""


def test_a_test_file_named_from_outside_the_repository_exits_2(project: Project) -> None:
    project.write("tests/test_rules.py", PROJECT_TESTS)

    result = project.weaken(str(project.root / "tests" / "test_rules.py"))

    assert result.returncode == 2
    assert "name a test file by its path inside the repository" in result.stderr
    assert result.stdout == ""


SWITCHED_OFF_TEST = """

def test_a_contract_that_refuses_nothing_is_still_counted(tmp_path):
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    assert (tmp_path / "pyproject.toml").read_text().count("[[tool.importlinter.contracts]]") == 1
"""


def test_a_test_that_only_counts_the_contracts_catches_nothing(project: Project) -> None:
    """A contract that refuses no import is still in the file, so a test of how many
    contracts there are passes, and the change survives until a test tries an import."""
    project.write(
        "tests/test_rules.py",
        PROJECT_TESTS.split("def test_low_importing_high")[0] + SWITCHED_OFF_TEST,
    )

    result = project.weaken("tests/test_rules.py", "--only", "no import refused")

    assert result.returncode == 1
    assert result.stdout == (
        "SURVIVED  pkg is layered: no import refused\n1 changes tried, 1 survived\n"
    )


def test_a_run_from_a_folder_without_pyproject_exits_2(project: Project) -> None:
    result = project.weaken(cwd=project.root / "src")

    assert result.returncode == 2
    assert "run this from the top of the repository" in result.stderr
