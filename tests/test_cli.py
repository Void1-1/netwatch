"""The `netwatch` command, run offline on synthetic data in a temp dir"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from netwatch import __version__
from netwatch.cli import app
from netwatch.config import example_config_text
from netwatch.models import Alert, AlertStatus
from netwatch.pipeline import WINDOWED_UNTIL_KEY
from netwatch.storage.db import Database
from netwatch.synthetic.generator import default_start, generate_normal_traffic

runner = CliRunner()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(*args: str) -> Result:
    return runner.invoke(app, list(args))


def test_version() -> None:
    result = run("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_init_creates_dirs_db_and_config(workdir: Path) -> None:
    result = run("init")
    assert result.exit_code == 0, result.output
    assert (workdir / "config.yaml").read_text() == example_config_text()
    assert (workdir / "data" / "netwatch.db").is_file()
    assert (workdir / "models").is_dir()

    (workdir / "config.yaml").write_text("# my edits\n")
    assert run("init").exit_code == 0
    assert (workdir / "config.yaml").read_text() == "# my edits\n"  # never overwritten


def test_invalid_config_is_a_clean_error(workdir: Path) -> None:
    (workdir / "config.yaml").write_text("scoring:\n  alert_treshold: 0.9\n")
    result = run("alerts")
    assert result.exit_code == 1
    assert "unknown key" in result.output


def test_init_with_a_broken_existing_config_is_a_clean_error(workdir: Path) -> None:
    (workdir / "config.yaml").write_text("scoring:\n  alert_treshold: 0.9\n")
    result = run("init")
    assert result.exit_code == 1
    assert "unknown key" in result.output


def test_zero_and_negative_options_are_rejected_not_ignored(workdir: Path) -> None:
    result = run("collect", "--once", "--interval", "0")
    assert result.exit_code == 1
    assert "--interval must be greater than 0" in result.output
    assert run("train", "--days", "0").exit_code != 0


def test_alerts_listing_survives_markup_in_process_names(workdir: Path) -> None:
    """process names and paths are chosen by whatever runs, malware included"""
    assert run("init").exit_code == 0
    name = "helper[/]"
    with Database.open(workdir / "data" / "netwatch.db") as db:
        db.insert_alert(
            Alert(1.0, name, 0.0, 1.0, {}, [f"process '{name}' was never seen during training"])
        )
        db.insert_alert(Alert(2.0, "[bold]x[/bold]", 0.0, 1.0, {}, ["runs from /tmp/[red]a"]))
    listing = run("alerts")
    assert listing.exit_code == 0, listing.output
    # shown exactly as they are, not interpreted
    assert name in listing.output
    assert "[bold]x[/bold]" in listing.output
    assert "/tmp/[red]a" in listing.output


def test_watch_without_a_model_explains_what_to_do(workdir: Path) -> None:
    result = run("watch")
    assert result.exit_code == 1
    assert "netwatch train" in result.output


def test_errors_are_not_broken_into_lines(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """an error names a path and a command to copy, however deep the working directory is"""
    deep = workdir / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    result = run("watch")
    assert result.exit_code == 1
    model_path = deep.resolve() / "models" / "netwatch-model.pkl"
    assert result.output == f"error: no trained model at {model_path}: run `netwatch train`\n"


def test_offline_flow_build_train_alerts_label(workdir: Path) -> None:
    """build-windows -> train -> alerts -> label on synthetic events"""
    assert run("init").exit_code == 0
    db_path = workdir / "data" / "netwatch.db"
    with Database.open(db_path) as db:
        db.insert_events(generate_normal_traffic(default_start(), days=3, seed=1))

    result = run("build-windows")
    assert result.exit_code == 0, result.output
    assert "Built" in result.output
    assert "Built 0 new windows" in run("build-windows").output  # nothing new to window

    result = run("train", "--days", "2")
    assert result.exit_code == 0, result.output
    assert "Trained on" in result.output
    assert (workdir / "models" / "netwatch-model.pkl").is_file()

    assert "No alerts yet" in run("alerts").output
    with Database.open(db_path) as db:
        window = db.query_windows()[-1]
        alert = Alert(
            window.window_end, window.process_name, window.window_start, 1.0, {}, ["test reason"]
        )
        alert_id = db.insert_alert(alert)

        db.attach_repeat(alert_id, window.window_start + 300)

    listing = run("alerts", "--limit", "5")
    assert listing.exit_code == 0
    assert "test reason" in listing.output
    assert "2x" in listing.output  # the alert and its repeat

    result = run("label", str(alert_id), "expected", "--note", "that was me")
    assert result.exit_code == 0, result.output
    assert "all 2 of its windows" in " ".join(result.output.split())
    with Database.open(db_path) as db:
        assert db.get_alert(alert_id).status == AlertStatus.EXPECTED
        assert db.list_labels()[0].note == "that was me"

    assert run("label", "999", "expected").exit_code == 1
    assert run("label", str(alert_id), "maybe").exit_code != 0

    # with a model in place, build-windows scores what it windows instead of trusting it
    # (as if the first build-windows had run right where the stored events ended, with another
    # day collected since)
    done = default_start() + 3 * 86_400
    with Database.open(db_path) as db:
        db.set_meta(WINDOWED_UNTIL_KEY, repr(done))
        db.insert_events(generate_normal_traffic(done, days=1, seed=2))
    result = run("build-windows")
    assert result.exit_code == 0, result.output
    assert "Built 0 new windows" not in result.output
    assert "Scored with the trained model" in result.output
    with Database.open(db_path) as db:
        assert float(db.get_meta(WINDOWED_UNTIL_KEY)) > done


def test_demo_refuses_to_reuse_a_database(workdir: Path) -> None:
    (workdir / "old.db").write_text("")
    result = run("demo", "--db", "old.db")
    assert result.exit_code == 1
    assert "exists" in result.output
