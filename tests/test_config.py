from pathlib import Path

from axial.config import load_config, write_default_config
from axial.store import Store


def test_defaults_match_the_spec(tmp_path: Path):
    cfg = load_config(Store(tmp_path))
    assert cfg.judge_model == "claude-opus-5"
    assert cfg.thresholds.min_tpr == 0.85
    assert cfg.thresholds.min_tnr == 0.85
    assert cfg.thresholds.max_gap == 0.10
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
