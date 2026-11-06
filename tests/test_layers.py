"""The import contracts in pyproject.toml, run by `lint-imports` in scripts/check.sh and CI.

Each test copies the package and the contracts to a temporary directory, edits the copy,
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
    """A copy of src/axial under tmp_path/src, with the contracts beside it."""
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


PACKAGES = "the packages are layered"
ANNOTATE = "only the cli imports the annotate app"
MEASURE = "measure and pairing import neither the alignment report nor threshold tuning"
STATS = "stats takes only traces from core"
ANNOTATE_APP = "from axial.analysis.annotate.app import create_app"


def add_import(package: Path, module: str, line: str) -> None:
    with (package / module).open("a") as f:
        f.write(f"\n{line}\n")


def test_the_package_as_it_stands_keeps_every_contract(package: Path) -> None:
    result = lint(package)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Contracts: 9 kept, 0 broken." in result.stdout


@pytest.mark.parametrize(
    ("module", "line"),
    [
        ("errors.py", "from axial.core.traces import Trace"),
        ("core/store.py", "from axial.stats.align import align"),
        ("stats/align.py", "from axial.analysis.dataset import Label"),
        ("analysis/dataset.py", "from axial.judge.spec import JudgeSpec"),
        ("judge/spec.py", "from axial.monitor.measure import measure_batch"),
        ("monitor/gate.py", "from axial.cli import app"),
    ],
)
def test_an_import_that_reaches_up_a_package_is_refused(
    package: Path, module: str, line: str
) -> None:
    """One import across each step of the package order, from the lower side."""
    add_import(package, module, line)

    result = lint(package)

    assert result.returncode == 1
    assert f"{PACKAGES} BROKEN" in result.stdout


@pytest.mark.parametrize(
    ("module", "line", "contract"),
    [
        ("monitor/pairing.py", "from axial.monitor.gate import gate", "monitor is layered"),
        (
            "monitor/measure.py",
            "from axial.monitor.pairing import pair_batches",
            "monitor is layered",
        ),
        # promotion imported nothing from the alignment report once the seams were fixed.
        (
            "judge/promotion.py",
            "from axial.judge.alignment import write_alignment",
            "judge is layered",
        ),
        ("judge/run.py", "from axial.judge.promotion import require_promoted", "judge is layered"),
        ("judge/spec.py", "from axial.judge.run import read_verdicts", "judge is layered"),
        ("analysis/dataset.py", ANNOTATE_APP, "analysis is layered"),
        ("analysis/sample.py", "from axial.analysis.dataset import Label", "analysis is layered"),
        ("stats/align.py", "from axial.stats.corrected import corrected_rate", "stats is layered"),
        ("core/config.py", "from axial.core.reviews import Review", "core is layered"),
        ("core/store.py", "from axial.core.config import Thresholds", "core is layered"),
        ("core/traces.py", "from axial.core.store import Store", "core is layered"),
    ],
)
def test_an_import_that_reaches_up_inside_a_package_is_refused(
    package: Path, module: str, line: str, contract: str
) -> None:
    """One import from each row to the row above it, so the order of every package's rows
    is pinned."""
    add_import(package, module, line)

    result = lint(package)

    assert result.returncode == 1
    assert f"{contract} BROKEN" in result.stdout
    assert f"{PACKAGES} KEPT" in result.stdout


@pytest.mark.parametrize(
    ("module", "line", "contract"),
    [
        # threshold imported the alignment report before the seams were fixed.
        (
            "judge/threshold.py",
            "from axial.judge.alignment import write_alignment",
            "judge is layered",
        ),
        (
            "judge/alignment.py",
            "from axial.judge.threshold import write_tuned_spec",
            "judge is layered",
        ),
        ("judge/promotion.py", "from axial.judge.checks import cost_usd", "judge is layered"),
        ("judge/checks.py", "from axial.judge.promotion import promote", "judge is layered"),
        (
            "analysis/dataset.py",
            "from axial.analysis.taxonomy import Taxonomy",
            "analysis is layered",
        ),
        ("analysis/taxonomy.py", "from axial.analysis.dataset import Label", "analysis is layered"),
        ("stats/align.py", "from axial.stats.perplexity import score_traces", "stats is layered"),
        ("stats/perplexity.py", "from axial.stats.align import align", "stats is layered"),
        ("core/config.py", "from axial.core.model import Model", "core is layered"),
        ("core/model.py", "from axial.core.config import Thresholds", "core is layered"),
    ],
)
def test_an_import_across_the_same_row_is_refused(
    package: Path, module: str, line: str, contract: str
) -> None:
    """Both directions for every pair of modules that share a row."""
    add_import(package, module, line)

    result = lint(package)

    assert result.returncode == 1
    assert f"{contract} BROKEN" in result.stdout
    assert f"{PACKAGES} KEPT" in result.stdout


@pytest.mark.parametrize(
    ("path", "contract"),
    [
        ("newmod.py", PACKAGES),
        ("monitor/newmod.py", "monitor is layered"),
        ("judge/newmod.py", "judge is layered"),
        ("analysis/newmod.py", "analysis is layered"),
        ("stats/newmod.py", "stats is layered"),
        ("core/newmod.py", "core is layered"),
    ],
)
def test_a_module_no_contract_places_is_refused(package: Path, path: str, contract: str) -> None:
    """A new file gets a row before it can import anything, beside the packages or inside
    one. Its only import here is one every module may make, so the missing row is all
    that is wrong with it."""
    (package / path).write_text("from axial.errors import AxialError\n")

    result = lint(package)

    assert result.returncode == 1
    assert f"{contract} BROKEN" in result.stdout
    assert "newmod" in result.stdout
    assert "Contracts: 8 kept, 1 broken." in result.stdout


@pytest.mark.parametrize(
    ("module", "line", "contract"),
    [
        ("monitor/gate.py", ANNOTATE_APP, ANNOTATE),
        ("judge/checks.py", ANNOTATE_APP, ANNOTATE),
        ("monitor/measure.py", "from axial.judge.alignment import write_alignment", MEASURE),
        ("monitor/measure.py", "from axial.judge.threshold import write_tuned_spec", MEASURE),
        ("monitor/pairing.py", "from axial.judge.alignment import write_alignment", MEASURE),
        ("monitor/pairing.py", "from axial.judge.threshold import write_tuned_spec", MEASURE),
        ("stats/align.py", "from axial.core.store import Store", STATS),
        ("stats/align.py", "from axial.core.config import Thresholds", STATS),
        ("stats/align.py", "from axial.core.model import Model", STATS),
        ("stats/align.py", "from axial.core.reviews import Review", STATS),
    ],
)
def test_an_import_the_package_order_allows_can_still_be_refused(
    package: Path, module: str, line: str, contract: str
) -> None:
    """Each of these goes from a higher package to a lower one, so only its own rule
    stands in the way."""
    add_import(package, module, line)

    result = lint(package)

    assert result.returncode == 1
    assert f"{contract} BROKEN" in result.stdout
    assert "Contracts: 8 kept, 1 broken." in result.stdout
