"""The layer contract in pyproject.toml, run by `lint-imports` in scripts/check.sh and CI.

Each test copies the package and the contract to a temporary directory, edits the copy,
and runs the real command on it, so the suite itself says what the check refuses.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LINT_IMPORTS = Path(sys.executable).with_name("lint-imports")


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """A copy of src/axial under tmp_path/src, with the contract beside it."""
    shutil.copytree(
        REPO / "src" / "axial",
        tmp_path / "src" / "axial",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    return tmp_path / "src" / "axial"


def lint(package: Path) -> subprocess.CompletedProcess[str]:
    root = package.parent.parent
    return subprocess.run(
        [str(LINT_IMPORTS), "--no-cache", "--config", "pyproject.toml"],
        cwd=root,
        env={"PATH": str(LINT_IMPORTS.parent), "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_package_as_it_stands_keeps_the_contract(package: Path) -> None:
    result = lint(package)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "axial is layered KEPT" in result.stdout


def test_an_import_that_reaches_up_is_refused(package: Path) -> None:
    with (package / "store.py").open("a") as f:
        f.write("\nfrom axial.judge import JudgeSpec\n")

    result = lint(package)

    assert result.returncode == 1
    assert "axial.store is not allowed to import axial.judge" in result.stdout


def test_an_import_across_the_same_row_is_refused(package: Path) -> None:
    """measure imported report before the seams were fixed; the contract keeps them apart."""
    with (package / "measure.py").open("a") as f:
        f.write("\nfrom axial.report import write_alignment\n")

    result = lint(package)

    assert result.returncode == 1
    assert "axial.measure is not allowed to import axial.report" in result.stdout


def test_a_module_the_contract_does_not_place_is_refused(package: Path) -> None:
    """A new file gets a row before it can import anything, so it cannot sit outside the rule."""
    (package / "newmod.py").write_text("from axial.cli import app\n")

    result = lint(package)

    assert result.returncode == 1
    assert "newmod" in result.stdout
