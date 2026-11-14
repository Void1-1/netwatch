"""netwatch: local network behaviour anomaly detection.

netwatch learns what "normal" network activity looks like for each process on this machine and
flags connections that do not fit. It is built as a pipeline of small stages, each in its own
subpackage:

    collector  -> snapshot live sockets and attribute each one to a process
    storage    -> persist events, feature windows, alerts and labels in SQLite
    features   -> turn raw events into per-process time windows of features
    detectors  -> layered anomaly detectors behind one common interface
    scoring    -> put detector scores on a common scale and combine
    alerts     -> deliver explained alerts to the terminal or desktop
    feedback   -> user labels, and the training-set rules that use them
    synthetic  -> realistic fake traffic and attacks for demo or testing

Everything runs and stays on the machine, and no network connections are made.
"""

__version__ = "0.1.0"
