from pathlib import Path

import pytest

from axial.config import load_config, write_default_config
from axial.errors import AxialError
from axial.store import Store


def test_defaults_match_the_spec(tmp_path: Path):
    cfg = load_config(Store(tmp_path))
    assert cfg.judge_model == "claude-opus-5"
    assert cfg.thresholds.min_tpr == 0.85
    assert cfg.thresholds.min_tnr == 0.85
    assert cfg.thresholds.max_gap == 0.10
    assert cfg.thresholds.min_test_positives == 30
    assert cfg.thresholds.min_test_negatives == 30
    assert cfg.min_open_codes == 30
    assert cfg.split_fractions == (0.15, 0.42, 0.43)


def test_written_config_is_read_back(tmp_path: Path):
    store = Store(tmp_path)
    write_default_config(store)
    assert store.config_path.exists()
    assert load_config(store).judge_model == "claude-opus-5"


def test_file_values_override_defaults(tmp_path: Path):
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text(
        'judge_model = "claude-sonnet-5"\nannotator = "andres"\n\n'
        "[thresholds]\nmin_tpr = 0.9\n"
    )
    cfg = load_config(store)
    assert cfg.judge_model == "claude-sonnet-5"
    assert cfg.annotator == "andres"
    assert cfg.thresholds.min_tpr == 0.9
    assert cfg.thresholds.min_tnr == 0.85  # untouched default
    assert cfg.thresholds.min_test_positives == 30  # untouched default
    assert cfg.thresholds.min_test_negatives == 30  # untouched default


def test_file_setting_only_min_test_positives_leaves_the_rest_at_default(tmp_path: Path):
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text("[thresholds]\nmin_test_positives = 50\n")
    cfg = load_config(store)
    assert cfg.thresholds.min_test_positives == 50
    assert cfg.thresholds.min_test_negatives == 30
    assert cfg.thresholds.min_tpr == 0.85
    assert cfg.thresholds.min_tnr == 0.85
    assert cfg.thresholds.max_gap == 0.10


def test_split_fractions_with_two_elements_raises(tmp_path: Path):
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text("split_fractions = [0.5, 0.5]")
    with pytest.raises(AxialError, match="exactly 3 entries"):
        load_config(store)


def test_split_fractions_with_four_elements_raises(tmp_path: Path):
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text("split_fractions = [0.25, 0.25, 0.25, 0.25]")
    with pytest.raises(AxialError, match="exactly 3 entries"):
        load_config(store)


def test_split_fractions_with_non_positive_entry_raises(tmp_path: Path):
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text("split_fractions = [0.1, 0.0, 0.9]")
    with pytest.raises(AxialError, match="must be positive"):
        load_config(store)


def test_split_fractions_not_summing_to_one_raises(tmp_path: Path):
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text("split_fractions = [0.3, 0.3, 0.3]")
    with pytest.raises(AxialError, match="sum to 1.0"):
        load_config(store)


def test_split_fractions_valid_default_still_loads(tmp_path: Path):
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text("split_fractions = [0.15, 0.42, 0.43]")
    cfg = load_config(store)
    assert cfg.split_fractions == (0.15, 0.42, 0.43)
