import os
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


# --- .env loading ------------------------------------------------------------


def test_load_env_reads_keys_from_a_dotenv_file(tmp_path, monkeypatch):
    from axial.config import load_env

    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-from-file\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    load_env(tmp_path)

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-file"


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    """The file is a convenience. An explicitly exported key is a deliberate act."""
    from axial.config import load_env

    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-from-file\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-exported")

    load_env(tmp_path)

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-exported"


def test_load_env_is_a_no_op_without_a_dotenv_file(tmp_path, monkeypatch):
    from axial.config import load_env

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert load_env(tmp_path) is None
    assert "OPENROUTER_API_KEY" not in os.environ


def test_load_env_ignores_comments_and_blank_lines(tmp_path, monkeypatch):
    from axial.config import load_env

    (tmp_path / ".env").write_text(
        "# the key Jev needs\n\nOPENROUTER_API_KEY=sk-or-real\n# ANTHROPIC_API_KEY=unset\n"
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    load_env(tmp_path)

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-real"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_an_empty_placeholder_value_is_not_treated_as_a_key(tmp_path, monkeypatch):
    """A freshly copied .env.example has empty values; they must not mask a real export."""
    from axial.config import load_env

    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    load_env(tmp_path)

    assert not os.environ.get("OPENROUTER_API_KEY")
