"""Synthetic demo ``netwatch demo``

The demo runs the real pipeline (storage, windowing, novelty memory, training with the poisoning
rule, live monitoring) on synthetic data. Doesn't need root, network or real traffic:

1. Generate several days of normal traffic for a few fake processes.
2. Inject the four reference attacks into the final ("live") days.
3. Store everything in an in-memory SQLite database.
4. Window the warm-up and training days, then train the detectors.
5. Replay the live days slot by slot through :class:`~netwatch.pipeline.Monitor`, the same
   loop ``netwatch watch`` runs.
6. Optionally, a simulated reviewer labels each alert as ``netwatch label`` would. This
   exercises the feedback loop.
7. Compare alerts with the truth: which attacks were caught, with what reasons, and how many
   alerts fired on normal windows.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from netwatch.alerts.console import format_time, render_alert
from netwatch.config import Config
from netwatch.features.novelty import NoveltyTracker
from netwatch.feedback.labels import record_label
from netwatch.models import Alert, Verdict
from netwatch.pipeline import (
    Monitor,
    ScoredWindow,
    TrainingSelection,
    build_pending_windows,
    make_window_builder,
    new_tracker,
    select_training_windows,
    train_model,
)
from netwatch.storage.db import IN_MEMORY, Database
from netwatch.synthetic.attacks import InjectedAttack, inject, standard_attack_suite
from netwatch.synthetic.generator import DEFAULT_PROFILES, default_start, generate_normal_traffic

DAY = 86_400
MIN_TEST_DAYS = 2  # the attack suite spans the first two live days


class SimulatedReviewer:
    """labels each alert as they appear, knows the correct answer"""

    def __init__(
        self, db: Database, tracker: NoveltyTracker, attacks: Sequence[InjectedAttack]
    ) -> None:
        self.db = db
        self.tracker = tracker
        self.attacks = list(attacks)
        self.expected = 0
        self.suspicious = 0

    def emit(self, alert: Alert) -> None:
        if alert.id is None:
            return
        # monitor raises an alert when windows close, timestamps are end of windows
        is_attack = any(
            a.overlaps(alert.process_name, alert.window_start, alert.timestamp)
            for a in self.attacks
        )
        verdict = Verdict.SUSPICIOUS if is_attack else Verdict.EXPECTED
        record_label(
            self.db,
            alert.id,
            verdict,
            note="demo reviewer",
            tracker=self.tracker,
            now=alert.timestamp,
        )
        if is_attack:
            self.suspicious += 1
        else:
            self.expected += 1


@dataclass(frozen=True, slots=True)
class AttackOutcome:
    """how the pipeline did against an injected attack

    attributes:
        attack: the injected attack
        windows: live windows containing traffic
        flagged: windows that alerted, repeats included
        alerts: alerts raised; the other flagged windows were repeats
        first_alert: alert covering the first flagged window
    """

    attack: InjectedAttack
    windows: int
    flagged: int
    alerts: int
    first_alert: Alert | None

    @property
    def detected(self) -> bool:
        return self.flagged > 0


@dataclass(frozen=True, slots=True)
class DemoReport:
    """everything measured by the demo"""

    warmup_days: int
    train_days: int
    test_days: int
    processes: list[str]
    total_events: int
    total_windows: int
    training: TrainingSelection
    test_windows: int
    normal_test_windows: int
    outcomes: list[AttackOutcome]
    false_alerts: list[Alert]
    false_repeats: int
    reviewed: bool
    labeled_expected: int
    labeled_suspicious: int
    elapsed_seconds: float

    @property
    def detected_count(self) -> int:
        return sum(o.detected for o in self.outcomes)

    @property
    def all_detected(self) -> bool:
        return self.detected_count == len(self.outcomes)

    @property
    def false_alert_rate(self) -> float:
        return len(self.false_alerts) / max(self.normal_test_windows, 1)

    @property
    def repeats(self) -> int:
        """alerting windows attached to an open alert"""
        return self.false_repeats + sum(o.flagged - o.alerts for o in self.outcomes)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """alerts matched with actuality

    attributes:
        outcomes: 1 per injected attack
        false_alerts: alerts raised by windows without attacks
        false_repeats: repeats of normal windows that alerted
        normal_windows: normal windows scored
    """

    outcomes: list[AttackOutcome]
    false_alerts: list[Alert]
    false_repeats: int
    normal_windows: int


def evaluate(scored: Sequence[ScoredWindow], attacks: Sequence[InjectedAttack]) -> Evaluation:
    """match alerts against actuality"""

    attack_windows: set[tuple[str, float]] = set()
    outcomes: list[AttackOutcome] = []
    for attack in attacks:
        matched = [
            s
            for s in scored
            if attack.overlaps(s.window.process_name, s.window.window_start, s.window.window_end)
        ]
        attack_windows.update(s.window.key for s in matched)
        flagged = [s for s in matched if s.alert is not None]
        outcomes.append(
            AttackOutcome(
                attack=attack,
                windows=len(matched),
                flagged=len(flagged),
                alerts=sum(not s.repeat for s in flagged),
                first_alert=flagged[0].alert if flagged else None,
            )
        )
    normal = [s for s in scored if s.window.key not in attack_windows]
    return Evaluation(
        outcomes=outcomes,
        false_alerts=[s.alert for s in normal if s.alert is not None and not s.repeat],
        false_repeats=sum(s.repeat for s in normal),
        normal_windows=len(normal),
    )


def run_demo(
    config: Config | None = None,
    *,
    train_days: int = 7,
    test_days: int = 2,
    seed: int = 7,
    review: bool = True,
    db_path: str | Path = IN_MEMORY,
) -> DemoReport:
    """run synthetic pipeline and measure detection quality

    arguments:
        config: Configuration
        train_days: days of normal traffic to train on
        test_days: live days replayed with attacks
        seed: seed for synthetic traffic
        review: label alerts as they arrive with :class:`SimulatedReviewer`
        db_path: where to keep demo db
    """
    if test_days < MIN_TEST_DAYS:
        raise ValueError(f"the demo needs at least {MIN_TEST_DAYS} live (test) days")
    config = config or Config()
    began = time.perf_counter()

    # 1 - 2. Synthetic traffic for warm-up, training, live days
    warmup_days = math.ceil(config.training.warmup_hours / 24)
    start = default_start()
    test_start = start + (warmup_days + train_days) * DAY
    normal = generate_normal_traffic(
        start,
        warmup_days + train_days + test_days,
        seed=seed,
        collector_interval=config.collector.interval_seconds,
    )
    injections = standard_attack_suite(test_start, seed=seed)
    events = inject(normal, injections)

    with Database.open(db_path) as db:
        # 3. store raw events
        db.insert_events(events)

        # 4. window warm-up, training days and train
        tracker = new_tracker(config)
        builder = make_window_builder(config)
        build_pending_windows(db, tracker, builder, until=test_start)
        selection = select_training_windows(
            db, days=train_days, warmup_hours=config.training.warmup_hours
        )
        model = train_model(selection.windows, config, now=test_start)

        # 5 - 6. replay live days through `watch`, optionally w/ reviewer
        attacks = [i.attack for i in injections]
        reviewer = SimulatedReviewer(db, tracker, attacks)
        monitor = Monitor(
            model,
            tracker,
            builder,
            config.scoring.alert_threshold,
            sinks=[reviewer] if review else [],
            db=db,
        )
        scored = monitor.process_events(db.query_events(start=test_start))
        total_windows = db.count_windows()

        # 7. score against truth
        evaluation = evaluate(scored, attacks)
        stored = {alert.id: alert for alert in db.query_alerts(start=test_start)}
        false_alerts = [stored.get(alert.id, alert) for alert in evaluation.false_alerts]
    return DemoReport(
        warmup_days=warmup_days,
        train_days=train_days,
        test_days=test_days,
        processes=[p.name for p in DEFAULT_PROFILES],
        total_events=len(events),
        total_windows=total_windows,
        training=selection,
        test_windows=len(scored),
        normal_test_windows=evaluation.normal_windows,
        outcomes=evaluation.outcomes,
        false_alerts=false_alerts,
        false_repeats=evaluation.false_repeats,
        reviewed=review,
        labeled_expected=reviewer.expected,
        labeled_suspicious=reviewer.suspicious,
        elapsed_seconds=time.perf_counter() - began,
    )


def render_report(report: DemoReport, console: Console) -> None:
    """print summary of :class:`DemoReport`"""
    console.print(
        Panel.fit(
            "Synthetic run: fake processes, injected attacks, and pipeline.\n"
            "No root or network access; database in memory unless --db is passed.",
            title="netwatch demo",
            border_style="cyan",
        )
    )

    training = report.training
    facts = Table.grid(padding=(0, 2))
    facts.add_row(
        "Simulated",
        f"{report.warmup_days}d warm-up + {report.train_days}d training "
        f"+ {report.test_days}d live monitoring",
    )
    facts.add_row("Processes", f"{len(report.processes)} fake: " + ", ".join(report.processes))
    facts.add_row("Events", f"{report.total_events:,} connections")
    facts.add_row("Windows", f"{report.total_windows:,} per-process windows")
    facts.add_row(
        "Training",
        f"{len(training.windows):,} windows "
        f"({training.excluded_alerted:,} excluded for potential poisoning)",
    )
    facts.add_row("Live", f"{report.test_windows:,} windows scored")
    if report.reviewed:
        facts.add_row(
            "Feedback",
            f"simulated reviewer labelled {report.labeled_suspicious:,} alerts suspicious and "
            f"{report.labeled_expected:,} expected",
        )
    else:
        facts.add_row(
            "Feedback",
            "none (--no-review): alerts stay open; repeats are attached to them",
        )
    console.print(facts)
    console.print()

    attacks = Table(title="Injected attacks")
    attacks.add_column("Attack", style="bold")
    attacks.add_column("Process")
    attacks.add_column("Started")
    attacks.add_column("Detected")
    attacks.add_column("Windows flagged", justify="right")
    attacks.add_column("Alerts", justify="right")
    for outcome in report.outcomes:
        attacks.add_row(
            escape(outcome.attack.kind),
            escape(outcome.attack.process_name),
            format_time(outcome.attack.start),
            "[green]yes[/]" if outcome.detected else "[red]MISSED[/]",
            f"{outcome.flagged}/{outcome.windows}",
            str(outcome.alerts),
        )
    console.print(attacks)

    console.print("\n[bold]Why attacks were flagged[/] (from first alert):")
    for outcome in report.outcomes:
        console.print(
            f"\n[bold]{escape(outcome.attack.kind)}[/] "
            f"[dim]- {escape(outcome.attack.description)}[/]"
        )
        if outcome.first_alert is None:
            console.print("  [red]not detected[/]")
            continue
        for reason in outcome.first_alert.reasons:
            console.print(f"  - {escape(reason)}")

    example = next((o.first_alert for o in report.outcomes if o.first_alert), None)
    if example is not None:
        console.print("\nHow a live alert looks with `netwatch watch`:")
        console.print(render_alert(example))

    console.print(
        f"\nFalse alerts: [bold]{len(report.false_alerts)}[/] of "
        f"{report.normal_test_windows:,} normal live windows ({report.false_alert_rate:.2%})"
    )
    if report.repeats:
        console.print(
            f"Repeats: [bold]{report.repeats}[/] alerting windows "
            f"({report.false_repeats} normal, {report.repeats - report.false_repeats} attack) "
            "only repeated the cause of an open alert; attached to it"
        )
    if report.false_alerts:
        false_table = Table(show_header=True, header_style="dim")
        false_table.add_column("Window")
        false_table.add_column("Process")
        false_table.add_column("Score", justify="right")
        false_table.add_column("Seen", justify="right")
        false_table.add_column("Top reason")
        for alert in report.false_alerts[:5]:
            false_table.add_row(
                format_time(alert.window_start),
                escape(alert.process_name),
                f"{alert.combined_score:.3f}",
                f"{alert.occurrences}x",
                escape(alert.reasons[0]) if alert.reasons else "-",
            )
        console.print(false_table)
        if len(report.false_alerts) > 5:
            console.print(f"[dim]...and {len(report.false_alerts) - 5} more[/]")

    total = len(report.outcomes)
    if report.all_detected:
        verdict = f"[bold green]All {total} injected attacks detected.[/]"
    else:
        verdict = f"[bold red]{report.detected_count} of {total} injected attacks detected.[/]"
    console.print(f"\n{verdict} [dim](ran in {report.elapsed_seconds:.1f}s)[/]")
