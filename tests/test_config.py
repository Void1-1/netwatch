"""Configuration: documented defaults, strict validation, path resolution"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from netwatch.config import Config, ConfigError, config_from_dict, example_config_text, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_config_documents_exactly_the_defaults() -> None:
    """config.example.yaml must always match the dataclass defaults"""
    example = config_from_dict(yaml.safe_load(example_config_text()))
    assert example == Config()


def test_example_config_text_is_the_repo_file() -> None:
    assert example_config_text() == (REPO_ROOT / "config.example.yaml").read_text()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"scoring": {"alert_treshold": 0.9}}, "unknown key"),
        ({"bogus": {}}, "unknown key"),
        ({"features": {"window_seconds": True}}, "integer"),
        ({"collector": {"interval_seconds": "fast"}}, "number"),
        ({"features": {"suspicious_paths": "/tmp"}}, "list"),
        ({"scoring": {"alert_threshold": 1.5}}, "alert_threshold"),
        ({"collector": {"kind": "bluetooth"}}, "collector.kind"),
        ({"detectors": {"autoencoder": {"enabled": True}}}, "not implemented"),
        ({"detectors": {"sequence": {"enabled": True}}}, "not implemented"),
        ({"detectors": {"novelty": {"weight": 1.1}}}, "novelty.weight"),
        ({"detectors": {"baseline": {"weight": -0.1}}}, "baseline.weight"),
        ({"detectors": {"sequence": {"weight": 2}}}, "sequence.weight"),
        ({"paths": []}, "mapping"),
    ],
)
def test_invalid_config_is_rejected(data: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_some_detector_must_be_able_to_raise_an_alert() -> None:
    """a percentile tops out at 1.0, so a weight below the alert threshold cannot alert alone"""
    names = ("novelty", "baseline", "isolation_forest")
    quiet = {"detectors": {name: {"weight": 0.99} for name in names}}
    with pytest.raises(ConfigError, match="nothing could ever raise an alert"):
        config_from_dict(quiet)
    # one detector at full weight is enough; the others may be turned down to explain-only
    quiet["detectors"]["baseline"]["weight"] = 1.0
    assert config_from_dict(quiet).detectors.novelty.weight == 0.99
    # a detector that is switched off does not count
    quiet["detectors"]["baseline"]["enabled"] = False
    with pytest.raises(ConfigError, match="nothing could ever raise an alert"):
        config_from_dict(quiet)


def test_ints_are_accepted_for_floats() -> None:
    assert (
        config_from_dict({"collector": {"interval_seconds": 2}}).collector.interval_seconds == 2.0
    )


def test_disabling_every_detector_is_rejected() -> None:
    data = {
        "detectors": {
            name: {"enabled": False} for name in ("novelty", "baseline", "isolation_forest")
        }
    }
    with pytest.raises(ConfigError, match="at least one detector"):
        config_from_dict(data)


def test_relative_paths_resolve_against_the_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    path = config_dir / "config.yaml"
    path.write_text("paths:\n  db_path: state/nw.db\n  models_dir: /abs/models\n")
    config = load_config(path)
    assert config.db_path == (config_dir / "state" / "nw.db").resolve()
    assert config.models_dir == Path("/abs/models")
    assert config.model_path.name == "netwatch-model.pkl"


def test_missing_config_falls_back_to_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config == Config()
    assert config.db_path == tmp_path.resolve() / "data" / "netwatch.db"


def test_explicit_missing_or_broken_files_fail(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("scoring: [unclosed\n")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(broken)


def test_empty_file_means_defaults(tmp_path: Path) -> None:
    empty = tmp_path / "config.yaml"
    empty.write_text("")
    assert load_config(empty) == Config()
