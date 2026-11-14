"""Fading first-seen memory for (process, ip) and (process, port) pairs"""

from __future__ import annotations

import pytest

from netwatch.features.novelty import SECONDS_PER_DAY, NoveltyTracker, PairKind

T0 = 1_800_000_000.0
DAY = SECONDS_PER_DAY


def test_unseen_pairs_are_novel_until_observed() -> None:
    tracker = NoveltyTracker()
    assert tracker.novel("app", PairKind.IP, ["192.0.2.1", "192.0.2.2"], T0) == [
        "192.0.2.1",
        "192.0.2.2",
    ]
    tracker.observe("app", PairKind.IP, ["192.0.2.1"], T0)
    assert tracker.novel("app", PairKind.IP, ["192.0.2.1", "192.0.2.2"], T0 + 60) == ["192.0.2.2"]


def test_lookups_never_mutate() -> None:
    tracker = NoveltyTracker()
    tracker.novel("app", PairKind.IP, ["192.0.2.1"], T0)
    tracker.is_known("app", PairKind.PORT, 443, T0)
    assert len(tracker) == 0


def test_memory_is_per_process() -> None:
    tracker = NoveltyTracker()
    tracker.observe("browser", PairKind.IP, ["192.0.2.1"], T0)
    assert tracker.is_known("browser", PairKind.IP, "192.0.2.1", T0)
    assert not tracker.is_known("editor", PairKind.IP, "192.0.2.1", T0)


def test_ports_and_ips_are_separate_kinds() -> None:
    tracker = NoveltyTracker()
    tracker.observe("app", PairKind.PORT, [443], T0)
    assert tracker.is_known("app", PairKind.PORT, 443, T0)
    assert tracker.novel("app", PairKind.PORT, [443, 8443], T0) == [8443]
    assert not tracker.is_known("app", PairKind.IP, "443", T0)


def test_single_sighting_fades_after_about_one_half_life() -> None:
    tracker = NoveltyTracker(half_life_days=14, min_familiarity=0.5)
    tracker.observe("app", PairKind.IP, ["192.0.2.1"], T0)
    assert tracker.familiarity("app", PairKind.IP, "192.0.2.1", T0 + 14 * DAY) == pytest.approx(0.5)
    assert tracker.is_known("app", PairKind.IP, "192.0.2.1", T0 + 13 * DAY)
    assert not tracker.is_known("app", PairKind.IP, "192.0.2.1", T0 + 15 * DAY)


def test_frequent_contacts_are_remembered_much_longer() -> None:
    tracker = NoveltyTracker(half_life_days=14)
    for day in range(30):
        tracker.observe("app", PairKind.IP, ["192.0.2.1"], T0 + day * DAY)
    tracker.observe("app", PairKind.IP, ["192.0.2.2"], T0 + 29 * DAY)
    later = T0 + 29 * DAY + 30 * DAY
    assert tracker.is_known("app", PairKind.IP, "192.0.2.1", later)
    assert not tracker.is_known("app", PairKind.IP, "192.0.2.2", later)


def test_observe_window_commits_ips_and_ports(make_window) -> None:
    tracker = NoveltyTracker()
    window = make_window("app", T0, remote_ips=("192.0.2.1",), remote_ports=(443,))
    tracker.observe_window(window)
    assert tracker.is_known("app", PairKind.IP, "192.0.2.1", window.window_end)
    assert tracker.is_known("app", PairKind.PORT, 443, window.window_end)


def test_prune_drops_only_faded_pairs() -> None:
    tracker = NoveltyTracker(half_life_days=1)
    tracker.observe("app", PairKind.IP, ["old"], T0)
    tracker.observe("app", PairKind.IP, ["fresh"], T0 + 10 * DAY)
    assert tracker.prune(T0 + 10 * DAY) == 1
    assert tracker.is_known("app", PairKind.IP, "fresh", T0 + 10 * DAY)


def test_records_round_trip() -> None:
    tracker = NoveltyTracker(half_life_days=7)
    tracker.observe("app", PairKind.IP, ["192.0.2.1"], T0)
    tracker.observe("app", PairKind.IP, ["192.0.2.1"], T0 + DAY)
    restored = NoveltyTracker.from_records(tracker.to_records(), half_life_days=7)
    at = T0 + 3 * DAY
    assert restored.familiarity("app", PairKind.IP, "192.0.2.1", at) == pytest.approx(
        tracker.familiarity("app", PairKind.IP, "192.0.2.1", at)
    )
    (record,) = restored.to_records()
    assert record.first_seen == T0
    assert record.last_seen == T0 + DAY


def test_merge_adds_pairs_and_keeps_the_more_familiar_side() -> None:
    ours = NoveltyTracker(half_life_days=7)
    ours.observe("app", PairKind.IP, ["shared", "ours"], T0)
    ours.observe("app", PairKind.IP, ["shared"], T0 + DAY)
    theirs = NoveltyTracker(half_life_days=7)
    theirs.observe("app", PairKind.IP, ["shared", "theirs"], T0 + 2 * DAY)
    at = T0 + 2 * DAY
    # seen twice by us outweighs seen once by them, even though theirs is newer
    expected = ours.familiarity("app", PairKind.IP, "shared", at)
    assert expected > theirs.familiarity("app", PairKind.IP, "shared", at)

    ours.merge_records(theirs.to_records())

    assert ours.known_processes() == {"app"}
    assert len(ours) == 3
    assert ours.familiarity("app", PairKind.IP, "shared", at) == pytest.approx(expected)
    assert ours.is_known("app", PairKind.IP, "theirs", at)
    shared = next(r for r in ours.to_records() if r.value == "shared")
    assert (shared.first_seen, shared.last_seen) == (T0, T0 + 2 * DAY)


def test_merge_is_order_free_and_never_double_counts() -> None:
    a, b = NoveltyTracker(), NoveltyTracker()
    a.observe("app", PairKind.IP, ["x", "y"], T0)
    b.observe("app", PairKind.IP, ["y", "z"], T0 + DAY)
    b.observe("app", PairKind.PORT, [443], T0 + DAY)

    ab = NoveltyTracker.from_records(a.to_records())
    ab.merge_records(b.to_records())
    ba = NoveltyTracker.from_records(b.to_records())
    ba.merge_records(a.to_records())
    assert sorted(ab.to_records()) == sorted(ba.to_records())

    # merging a stale copy back in, as every save does, changes nothing
    snapshot = ab.to_records()
    ab.observe("app", PairKind.IP, ["x"], T0 + 2 * DAY)
    before = sorted(ab.to_records())
    ab.merge_records(snapshot)
    ab.merge_records(b.to_records())
    assert sorted(ab.to_records()) == before


def test_rejects_non_positive_half_life() -> None:
    with pytest.raises(ValueError):
        NoveltyTracker(half_life_days=0)


def test_forget_drops_only_pairs_last_seen_up_to_the_cutoff() -> None:
    tracker = NoveltyTracker()
    tracker.observe("app", PairKind.IP, ["192.0.2.1", "192.0.2.2"], at=100.0)
    tracker.observe("app", PairKind.IP, ["192.0.2.2"], at=300.0)
    tracker.observe("other", PairKind.IP, ["192.0.2.1"], at=100.0)

    assert tracker.forget("app", PairKind.IP, ["192.0.2.1", "192.0.2.2", "192.0.2.3"], 200.0) == 1
    assert not tracker.is_known("app", PairKind.IP, "192.0.2.1", 300.0)
    assert tracker.is_known("app", PairKind.IP, "192.0.2.2", 300.0)  # seen again after the cutoff
    assert tracker.is_known("other", PairKind.IP, "192.0.2.1", 300.0)  # another process
    assert tracker.forget("app", PairKind.IP, ["192.0.2.1"], 200.0) == 0  # safe to repeat
