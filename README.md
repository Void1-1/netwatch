# netwatch

**A local network behaviour anomaly detector.**

netwatch learns what "normal" network activity looks like for *each process* on your machine, then flags connections that do not fit: a chat app suddenly checking in with a new server every 30 seconds, a mail client opening sixty connections to an unknown host at 3 a.m., a binary in `/tmp` phoning home. Every alert says, plainly, *why* it was flagged.

Everything runs and stays on your machine. netwatch makes no network calls of its own, and nothing it records is ever uploaded.

> Status: early alpha. The end-to-end pipeline works and is tested; the fancier detector layers are stubs (see the [roadmap](#roadmap-in-active-development)).

---

## Why process attribution matters

Most home-grade network monitoring looks at a machine as a whole: "this laptop talked to 300 hosts today". That total hides the interesting part. Port 443 to a new IP is normal for a web browser, but worrying for a text editor; a connection every 60 seconds is how a sync client works and how malware beacons.

So netwatch connects **every connection to the process that made it** (name, executable path, parent) and models behavior **per process**:

- each process gets its own baseline, its own novelty memory and, given enough history, its own model
- a busy browser's noise never raises the bar for a quiet daemon
- reasons are specific: *"first connection from this process to 203.0.113.7"*, not *"unusual traffic"*

## How it works

![netwatch pipeline: the collector feeds SQLite, events become per-process windows checked against novelty memory, the detectors score them, the combiner and deduplication decide on an alert, and your labels feed the poisoning rule that controls training and novelty memory](docs/architecture.drawio.svg)

<sub>The diagram is a [draw.io](https://app.diagrams.net) file: open [`docs/architecture.drawio.svg`](docs/architecture.drawio.svg) there (or with the draw.io editor extension) to edit it.</sub>

**Windows** Raw connections are grouped into 5-minute windows per process. Each window gets a handful of interpretable features: connection count, distinct remote IPs and ports, how many of those are *new* for this process, time of day, how regular repeated connections to the same peer are (beaconing), and if the executable lives somewhere suspicious like `/tmp`.

**Layered detectors** Every detector implements the same interface: `fit(windows)`, `score(windows)` (higher = more anomalous) and `explain(window)` (plain-language reasons).

| Layer | Detector | Looks for | Status |
| --- | --- | --- | --- |
| 1 | Novelty | New destinations, weighted by how unusual novelty is *for this process* (Poisson surprise vs. learned base rate) | done |
| 1 | Baseline | Volume, fan-out and timing far outside the process's own range (robust z-scores): spikes, scans, clockwork beacons | done |
| 2 | Isolation forest | Unusual *combinations* of window features, one forest per process | done |
| 3 | Autoencoder | Correlations between features (reconstruction error) | stub |
| 4 | Sequence model | Broken habits in the order of destinations; long-period beacons | stub |

Why two simple detectors before the forest: an isolation forest only cuts within the range it saw in training, so a window *far* beyond that range scores about the same as the most extreme normal window. The baseline detector measures *how* out of character a window is, and both layers share per-process baselines to explain themselves.

**Combining** Raw scores from different detectors are not comparable, so each is converted to a percentile against that detector's training history for the same process. The final score is a weighted *maximum*, because an attack only one layer found cannot be lost. Only detectors that contributed have their reasons shown. A detector's weight (0 to 1) is a switch more than a dial: below the alert threshold it can no longer raise an alert by itself and only adds its reasons to alerts raised by other detectors.

**Poisoning protection** A detector retrained on recent data could learn a long-lasting threat to be normal. So any window that raised an alert is excluded from training, and its destinations are not committed to novelty memory, unless its alert is labelled `expected`. The latest label wins: relabelling an alert `suspicious` takes its windows back out of training and makes the destinations it flagged unfamiliar again.

However, until labelled, a legitimately new destination stays new, and every revisit would alert. To solve this, netwatch **deduplicates**: a window whose new destinations are a subset of an *open* alert's, and whose other reasons (volume, fan-out, timing) that alert already covers, is attached to it as a repeat instead of raising a new alert. Reasons are compared by kind, not word for word, so 47 connections followed by 52 is the same news. A reason that grows to more than double what the alert reported is news again and raises its own alert.

Repeats don't notify, but they still count as alerted: they stay out of training and novelty memory, and `netwatch alerts` shows how many windows each alert covers and when it was last seen. The `expected` label settles the alert, re-admitting every covered window and recording its destinations. Anything new still raises its own alert, and so does a repeat after you label an alert `suspicious`. An open alert keeps absorbing repeats for as long as it stays open, so a large count in `netwatch alerts` deserves a look.

## Demo (no root or real traffic)

```bash
git clone https://github.com/Void1-1/netwatch.git
cd netwatch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
netwatch demo
```

`netwatch demo` generates ten days of realistic traffic for six fake processes (a browser, mail client, chat app, sync daemon, nightly updater, and occasional Python dev tool). The first day warms up the novelty memory, the next seven are training data, and the last two are replayed through the same live monitoring loop `netwatch watch` uses, with four attacks:

| Attack | Visible Difference |
| --- | --- |
| Periodic beacon | `chatapp` checks in with a new host every ~30 s for two hours |
| Large upload | `mailclient` opens 60 connections to a new host at 03:20 |
| `/tmp` dropper | an unknown binary in `/tmp/.cache` calls home on port 8443 |
| Port scan | `python3` probes 150 ports on a LAN host in 90 seconds |

It reports which were detected, with the reasons from the first alert, and how many alerts fired on normal windows (generally only a few of ~1,900, ~0.1-0.3%). A simulated reviewer labels each alert as it arrives, similarly to `netwatch label`. Pass `--no-review` for unattended behaviour: alerts stay open, and windows that only repeat an open alert's cause are attached rather than alerting again. Everything happens in an in-memory database; nothing is written to disk. Every synthetic remote address comes from a range reserved for documentation and benchmarking (RFC 5737, RFC 2544); the machine's own address and the port-scan target are private LAN addresses (RFC 1918). None of them is routable on the internet.

## Usage

```bash
netwatch init                     # data/, models/, database, config.yaml
sudo netwatch collect             # leave running for an extended period of time
netwatch build-windows            # raw events -> feature windows (scored too, once a model exists)
netwatch train --days 7           # fit detectors (honors the poisoning rule)
sudo netwatch watch               # live: collect, window, score, alert
netwatch alerts --limit 20        # review alerts, their reasons and repeats
netwatch label 12 expected --note "nightly backup to a new server"
netwatch label 13 suspicious
```

- **Root:** On macOS, seeing each process's sockets needs root privileges. Without them, netwatch falls back to the processes your user owns. Files created with `sudo` are handed back to the user, so later unprivileged commands keep working. On Linux, root is only needed to see other users' processes.
- **Warm-up:** The first 24 hours of collection are skipped for training: novelty memory starts empty, so everything looks new at first.
- **Config:** `config.yaml` (created by `init`) controls window size, thresholds, detector weights and more. Options are documented in [`config.example.yaml`](config.example.yaml).
- **Collection model:** The collector diffs periodic socket snapshots, so connections that open and close between two snapshots can be missed. Lower `collector.interval_seconds` if that becomes a problem. Connections that are already open when `collect` or `watch` starts are not recorded: only ones seen opening.
- **Nothing skips the model:** once a model exists, every stored event is scored before it can be learned from. `watch` scores each closed window from the database, so it also covers events that were collected but never windowed and events a separate `collect` stores while it runs. `build-windows` scores what it windows. A window is scored a few seconds after it ends, to let the last events arrive.
- **`collect` beside `watch`:** `watch` collects by itself, so `collect` is not needed while it runs. If both run on the same database, a connection they both see is stored once, and `watch` also scores what only `collect` saw (with `sudo`, it sees every user's processes).

## Privacy notes

- **Data stays local.** Events, windows, alerts, labels and models are stored in `data/` and `models/` locally. netwatch makes no outbound network calls, and the tests block socket connections to prove this.
- **Never publicize `data/` or `models/`.** Both are gitignored, so are `*.db`, `*.sqlite`, `*.mmdb`, `.env` and `config.yaml`. A trained model is derived from your traffic and could reveal activity.
- **Only monitor your own machines.** Use netwatch on machines and networks you own or are explicitly authorised to monitor. This project does not condone illegal usage.
- **Model files are trusted input.** Models are saved with `pickle` (see the [Python documentation](https://docs.python.org/3/library/pickle.html)), and loading one runs code from it. Only load model files you trained yourself.
- **`sudo` runs your files as root.** `sudo netwatch watch` runs the `netwatch` in your virtual environment and loads the model in `models/` with root privileges. Both normally belong to your user, so anything that can write to them as you can become root the next time you start it. netwatch refuses a model file that belongs to another user or that everyone can write, but it cannot protect files your own account controls. Keep the environment and `models/` writable by you alone, or install netwatch and keep its models somewhere only root can write.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

The test suite does not require root or network connections; GitHub CI runs on macOS and Linux with Python 3.11 and 3.12.

```
src/netwatch/
  collector/   psutil snapshot collector with process attribution
  storage/     SQLite schema and access layer
  features/    windowing, encoders, time-decayed novelty memory
  detectors/   common Detector ABC; novelty, baseline, isolation forest; stubs for layers 3-4
  scoring/     percentile normalisation and weighted-max combiner
  alerts/      deduplication of repeat alerts; terminal and macOS notification sinks
  feedback/    labels and the poisoning rule
  synthetic/   normal-traffic generator and attack injectors
  pipeline.py  wires the stages together (shared by the CLI and the demo)
  demo.py      the synthetic end-to-end evaluation
  cli.py       the `netwatch` command
```

## Roadmap (In Active Development)

- **Layer 3: autoencoder** detector on window features (reconstruction error, per-feature explanations)
- **Layer 4: sequence model** of each process's destination order, including beacons too slow to see inside one window.
- **DNS name capture**, so reasons can cite `updates.example.com`, not an IP
- **Byte counts**, for real upload/exfiltration volume over just connection counts
- **GeoIP / ASN enrichment** from a local MaxMind database ([`scripts/download_geoip.sh`](scripts/download_geoip.sh) is stubbed)
- **Local dashboard** for reviewing and labelling alerts
- Later: launchd/systemd service files, eBPF or packet-level capture

## License

[MIT](LICENSE)
