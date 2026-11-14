-- netwatch SQLite schema
--
-- One local database file holds everything netwatch knows. It lives under data/ (gitignored)
-- and never leaves the machine. Every statement is safe to repeat (idempotent), because the
-- schema is applied on every startup.

-- raw observations: one row per new connection the collector saw
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY,
    timestamp    REAL    NOT NULL,   -- epoch seconds
    pid          INTEGER,
    process_name TEXT    NOT NULL,
    exe_path     TEXT,
    parent_name  TEXT,
    local_ip     TEXT    NOT NULL,
    local_port   INTEGER NOT NULL,
    remote_ip    TEXT    NOT NULL,
    remote_port  INTEGER NOT NULL,
    protocol     TEXT    NOT NULL,
    status       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);
CREATE INDEX IF NOT EXISTS idx_events_process_name ON events (process_name);

-- per-process feature windows. Features and context are JSON, so a new feature needs no
-- migration
CREATE TABLE IF NOT EXISTS windows (
    id           INTEGER PRIMARY KEY,
    process_name TEXT NOT NULL,
    window_start REAL NOT NULL,
    window_end   REAL NOT NULL,
    features     TEXT NOT NULL,      -- JSON object, see WindowFeatures
    context      TEXT NOT NULL,      -- JSON object, see FeatureWindow.context_dict
    UNIQUE (process_name, window_start)
);
CREATE INDEX IF NOT EXISTS idx_windows_process_start ON windows (process_name, window_start);

-- alerts. (process_name, window_start) points back at the window, which is how the poisoning
-- rule keeps it out of training
CREATE TABLE IF NOT EXISTS alerts (
    id                  INTEGER PRIMARY KEY,
    timestamp           REAL NOT NULL,
    process_name        TEXT NOT NULL,
    window_start        REAL NOT NULL,
    combined_score      REAL NOT NULL,
    per_detector_scores TEXT NOT NULL,   -- JSON object: detector -> percentile
    reasons             TEXT NOT NULL,   -- JSON array of strings
    status              TEXT NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open', 'expected', 'suspicious'))
);
CREATE INDEX IF NOT EXISTS idx_alerts_window ON alerts (process_name, window_start);

-- later windows attached to an open alert because they only repeat its cause (same process, no
-- new destination, no new reason) instead of raising an alert of their own. They belong to the
-- alert's process, so (alerts.process_name, window_start) points back at each one. The poisoning
-- rule treats them like the alert's own window, and labelling the alert labels them too
CREATE TABLE IF NOT EXISTS alert_repeats (
    alert_id     INTEGER NOT NULL REFERENCES alerts (id),
    window_start REAL    NOT NULL,
    PRIMARY KEY (alert_id, window_start)
);

-- user verdicts, append-only. The latest label per alert wins
CREATE TABLE IF NOT EXISTS labels (
    id        INTEGER PRIMARY KEY,
    alert_id  INTEGER NOT NULL REFERENCES alerts (id),
    verdict   TEXT    NOT NULL CHECK (verdict IN ('expected', 'suspicious')),
    timestamp REAL    NOT NULL,
    note      TEXT
);
CREATE INDEX IF NOT EXISTS idx_labels_alert ON labels (alert_id);

-- novelty memory: which (process, ip) and (process, port) pairs are familiar. Stored so that
-- "first seen" survives restarts
CREATE TABLE IF NOT EXISTS novelty (
    process_name TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('ip', 'port')),
    value        TEXT NOT NULL,
    familiarity  REAL NOT NULL,
    last_seen    REAL NOT NULL,
    first_seen   REAL NOT NULL,
    PRIMARY KEY (process_name, kind, value)
);

-- small key/value store, e.g. how far events have been windowed
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
