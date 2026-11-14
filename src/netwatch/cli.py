"""CLI

Typical flow::

    netwatch init                 # once
    sudo netwatch collect         # for a few days
    netwatch build-windows
    netwatch train
    sudo netwatch watch           # live: collect, window, score, alert
    netwatch alerts
    netwatch label 12 expected --note "that was a backup job"

``netwatch demo`` runs the process on synthetic data without privileges
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.table import Table

from netwatch import __version__
from netwatch.alerts import build_sinks
from netwatch.alerts.console import format_time
from netwatch.collector.base import CollectorPermissionError
from netwatch.collector.psutil_collector import PsutilCollector
from netwatch.config import (
    DEFAULT_CONFIG_FILENAME,
    Config,
    ConfigError,
    example_config_text,
    load_config,
)
from netwatch.feedback.labels import record_label
from netwatch.models import Verdict
from netwatch.pipeline import (
    Monitor,
    PipelineError,
    build_pending_windows,
    load_model,
    load_tracker,
    make_window_builder,
    save_model,
    score_closed_slots,
    score_pending_events,
    select_training_windows,
    store_new_connections,
    sync_tracker,
    train_model,
)
from netwatch.storage.db import Database, hand_back_to_sudo_user

app = typer.Typer(
    name="netwatch",
    help="Watch each process's network connections on this machine and flag unusual ones.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help=f"Config file (default: ./{DEFAULT_CONFIG_FILENAME} if present, else the defaults).",
    ),
]


def _fail(message: str) -> typer.Exit:
    """Print an error and return an Exit to raise"""
    console.print(f"[bold red]error:[/] {escape(message)}")
    return typer.Exit(code=1)


def _interval(config: Config, interval: float | None) -> float:
    """seconds between snapshots: ``--interval`` if given, else the configured one"""
    if interval is None:
        return config.collector.interval_seconds
    if interval <= 0:
        raise _fail("--interval must be greater than 0")
    return interval


def _config(ctx: typer.Context) -> Config:
    try:
        return load_config(ctx.obj)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"netwatch {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: ConfigOption = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logging.")] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Watch each process's network connections on this machine and flag unusual ones."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    ctx.obj = config


# Setup and collection


@app.command()
def init(ctx: typer.Context) -> None:
    """create data/, models/, database, config.yaml"""
    config_path: Path = ctx.obj or Path.cwd() / DEFAULT_CONFIG_FILENAME
    if config_path.exists():
        console.print(f"Keeping existing config: {escape(str(config_path))}")
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(example_config_text(), encoding="utf-8")
        console.print(f"Wrote {escape(str(config_path))}")

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    for directory in (config.data_dir, config.models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with Database.open(config.db_path):
        pass
    console.print(f"Database ready: {escape(str(config.db_path))}")
    console.print("Next: [bold]sudo netwatch collect[/] (or check with [bold]netwatch demo[/])")


@app.command()
def collect(
    ctx: typer.Context,
    interval: Annotated[float | None, typer.Option(help="Seconds between snapshots")] = None,
    once: Annotated[bool, typer.Option(help="Take one snapshot and exit")] = False,
) -> None:
    """snapshot connections in a loop and store the new ones

    Connections already open when the loop starts are not recorded: only ones seen opening.
    `--once` records every connection open right now instead
    """
    config = _config(ctx)
    interval = _interval(config, interval)
    collector = PsutilCollector(
        kind=config.collector.kind, allow_partial=config.collector.allow_partial
    )
    total = 0
    with Database.open(config.db_path) as db:
        hand_back_to_sudo_user(config.data_dir, config.db_path)
        console.print(f"Collecting every {interval:g}s into {escape(str(config.db_path))}")
        try:
            if not once:
                collector.prime()
            while True:
                total += store_new_connections(db, collector, interval)
                if once:
                    break
                console.print(f"\r{total:,} connections recorded", end="")
                time.sleep(interval)
        except CollectorPermissionError as exc:
            raise _fail(str(exc)) from exc
        except KeyboardInterrupt:
            pass
    if collector.partial:
        console.print(
            "\n[yellow]Partial coverage:[/] only processes under the active user were visible."
        )
    console.print(f"\nRecorded {total:,} new connections.")


# Features and training


@app.command("build-windows")
def build_windows(ctx: typer.Context) -> None:
    """turn stored raw events into per-process feature windows

    Before a model exists the windows are only built: they are the training data. Once
    `netwatch train` has run they are also scored, exactly as `netwatch watch` would have, so
    events collected without `watch` still raise their alerts and cannot slip into novelty
    memory or the next training run unseen
    """
    config = _config(ctx)
    builder = make_window_builder(config)
    scored = None
    with Database.open(config.db_path) as db:
        tracker = load_tracker(db, config)
        if config.model_path.is_file():
            try:
                model = load_model(config.model_path)
            except PipelineError as exc:
                raise _fail(str(exc)) from exc
            monitor = Monitor(model, tracker, builder, config.scoring.alert_threshold, db=db)
            scored = score_pending_events(db, monitor)
            windows = [s.window for s in scored]
        else:
            windows = build_pending_windows(db, tracker, builder)
        sync_tracker(db, tracker)
        total = db.count_windows()
    processes = len({w.process_name for w in windows})
    console.print(
        f"Built {len(windows):,} new windows across {processes} processes "
        f"({total:,} windows stored)."
    )
    if scored is not None:
        raised = sum(1 for s in scored if s.alert is not None and not s.repeat)
        repeats = sum(1 for s in scored if s.repeat)
        console.print(
            f"Scored with the trained model. New alerts: {raised:,}. Repeats of open alerts: "
            f"{repeats:,} (see `netwatch alerts`)."
        )


@app.command()
def train(
    ctx: typer.Context,
    days: Annotated[
        int | None, typer.Option(min=1, help="Days of recent windows to train on")
    ] = None,
) -> None:
    """fit detectors on recent windows and save"""
    config = _config(ctx)
    days = config.training.days if days is None else days
    with Database.open(config.db_path) as db:
        try:
            selection = select_training_windows(db, days, config.training.warmup_hours)
            model = train_model(selection.windows, config)
        except PipelineError as exc:
            raise _fail(str(exc)) from exc
    save_model(model, config.model_path)

    table = Table.grid(padding=(0, 2))
    table.add_row("Period", f"{format_time(selection.start)} to {format_time(selection.end)}")
    table.add_row("Windows considered", f"{selection.considered:,}")
    table.add_row("Skipped (warm-up)", f"{selection.excluded_warmup:,}")
    table.add_row("Excluded (alerted, not labelled expected)", f"{selection.excluded_alerted:,}")
    table.add_row("Re-admitted (labelled expected)", f"{selection.readmitted_expected:,}")
    table.add_row(
        "Trained on", f"{model.training_windows:,} windows, {len(model.processes)} processes"
    )
    table.add_row("Detectors", ", ".join(d.name for d in model.detectors))
    table.add_row("Saved to", escape(str(config.model_path)))
    console.print(table)


# monitoring and review


@app.command()
def watch(
    ctx: typer.Context,
    interval: Annotated[float | None, typer.Option(help="Seconds between snapshots")] = None,
) -> None:
    """collect, window each closed slot, score, and emit alerts

    Every stored event of a closed slot is scored, not only the ones this command collected:
    events collected after the last `build-windows`, left over from an earlier run, or
    stored by a `netwatch collect` running at the same time. Nothing skips the model
    """
    config = _config(ctx)
    interval = _interval(config, interval)
    try:
        model = load_model(config.model_path)
    except PipelineError as exc:
        raise _fail(str(exc)) from exc

    builder = make_window_builder(config)
    collector = PsutilCollector(
        kind=config.collector.kind, allow_partial=config.collector.allow_partial
    )
    with Database.open(config.db_path) as db:
        hand_back_to_sudo_user(config.data_dir, config.db_path)
        tracker = load_tracker(db, config)
        monitor = Monitor(
            model,
            tracker,
            builder,
            config.scoring.alert_threshold,
            sinks=build_sinks(config.alerts, console),
            db=db,
        )
        console.print(
            f"Watching {len(model.processes)} known processes. Windows close every "
            f"{builder.window_seconds}s"
        )
        try:
            # sockets that are already open are not new connections
            collector.prime()
            # catch up on what was stored while nothing was scoring
            backlog = score_closed_slots(db, monitor, tracker)
            if backlog:
                console.print(f"Caught up on {len(backlog):,} windows stored earlier.")
            while True:
                store_new_connections(db, collector, interval)
                # scored from the database, so what another command stored is included
                score_closed_slots(db, monitor, tracker)
                time.sleep(interval)
        except CollectorPermissionError as exc:
            raise _fail(str(exc)) from exc
        except KeyboardInterrupt:
            console.print("\nStopped.")
        finally:
            sync_tracker(db, tracker)


@app.command()
def alerts(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option(help="how many recent alerts to show")] = 20,
) -> None:
    """list recent alerts with their reasons.

    "Seen" counts the windows an alert covers: the one that raised it and later repeats of the
    same cause
    """
    config = _config(ctx)
    with Database.open(config.db_path) as db:
        recent = db.list_alerts(limit)
    if not recent:
        console.print("No alerts yet.")
        return
    table = Table(show_lines=True)
    table.add_column("ID", justify="right")
    table.add_column("Window")
    table.add_column("Process", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Seen", justify="right")
    table.add_column("Status")
    table.add_column("Reasons")
    for alert in recent:
        window = format_time(alert.window_start)
        if alert.occurrences > 1:
            window += f"\nlast {format_time(alert.last_window_start)}"
        table.add_row(
            str(alert.id),
            window,
            # process names and paths come from whatever is running: never read them as markup
            escape(alert.process_name),
            f"{alert.combined_score:.3f}",
            f"{alert.occurrences}x",
            str(alert.status),
            escape("\n".join(f"- {r}" for r in alert.reasons)),
        )
    console.print(table)


@app.command()
def label(
    ctx: typer.Context,
    alert_id: Annotated[int, typer.Argument(help="Alert ID (see `netwatch alerts`)")],
    verdict: Annotated[Verdict, typer.Argument(help="expected or suspicious")],
    note: Annotated[str | None, typer.Option(help="note stored with the label")] = None,
) -> None:
    """label an alert and its repeats

    `expected` makes its destinations familiar and lets its windows be trained on. `suspicious`
    keeps them out, and takes back an earlier `expected`
    """
    config = _config(ctx)
    with Database.open(config.db_path) as db:
        tracker = load_tracker(db, config)
        try:
            record_label(db, alert_id, verdict, note=note, tracker=tracker)
        except KeyError as exc:
            raise _fail(f"no alert with id {alert_id}") from exc
        sync_tracker(db, tracker)
        alert = db.get_alert(alert_id)
    covered = 1 if alert is None else alert.occurrences
    windows = "its window" if covered == 1 else f"all {covered} of its windows"
    if verdict == Verdict.EXPECTED:
        console.print(
            f"Alert {alert_id} labelled expected: its destinations are considered safe and "
            f"{windows} will be used for the next `netwatch train`"
        )
    else:
        stay = "stays" if covered == 1 else "stay"
        console.print(
            f"Alert {alert_id} labelled suspicious: {windows} {stay} out of training and its "
            "new destinations stay unfamiliar."
        )


# Demo


@app.command()
def demo(
    ctx: typer.Context,
    train_days: Annotated[int, typer.Option(min=2, help="days of normal training data")] = 7,
    test_days: Annotated[
        int, typer.Option(min=2, help="days of live monitoring with attacks injected")
    ] = 2,
    seed: Annotated[int, typer.Option(help="random seed for the synthetic traffic")] = 7,
    review: Annotated[
        bool,
        typer.Option(
            help="Simulate a user labelling each alert as it arrives "
            "(--no-review for unattended behavior)"
        ),
    ] = True,
    db: Annotated[
        Path | None,
        typer.Option(help="write the demo database to this file instead of memory"),
    ] = None,
) -> None:
    """fully synthetic run: train, inject attacks, detect, report"""
    # imported lazily
    from netwatch.demo import render_report, run_demo

    config = _config(ctx)
    if db is not None and db.exists():
        raise _fail(f"{db} exists. Pick a new path")
    with console.status("Generating traffic, training, and replaying live days..."):
        report = run_demo(
            config,
            train_days=train_days,
            test_days=test_days,
            seed=seed,
            review=review,
            db_path=db if db is not None else ":memory:",
        )
    render_report(report, console)


if __name__ == "__main__":
    app()
