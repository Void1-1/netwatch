"""Configuration: defaults, YAML loading, validation

Every setting lives in the dataclasses below. Their defaults are the documented defaults:
``config.example.yaml`` must match them exactly (a test checks this), and a missing or empty
``config.yaml`` behaves the same as the example.

Loading is strict: unknown keys and wrongly typed values raise :class:`ConfigError`. In a security
tool a typo that is silently ignored (say ``alert_treshold``) is worse than a crash at startup.

Relative paths are resolved against the config file's directory, so
``netwatch --config ~/netwatch/config.yaml`` works from any directory
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

DEFAULT_CONFIG_FILENAME = "config.yaml"
EXAMPLE_CONFIG_FILENAME = "config.example.yaml"
MODEL_FILENAME = "netwatch-model.pkl"

# psutil's accepted `kind` values of net_connections()
_PSUTIL_KINDS = frozenset({"inet", "inet4", "inet6", "tcp", "tcp4", "tcp6", "udp", "udp4", "udp6"})


class ConfigError(ValueError):
    """raised for any invalid configuration: bad YAML, unknown keys, bad values"""


def _require(condition: bool, message: str) -> None:
    """raise :class:`ConfigError` with ``message`` unless ``condition`` is met"""
    if not condition:
        raise ConfigError(message)


def _require_weight(weight: float, where: str) -> None:
    """a detector weight scales a percentile, so it has to stay within [0, 1]

    Above 1 it would push percentiles that are *below* the alert threshold over it
    """
    _require(0 <= weight <= 1, f"{where} must be within [0, 1]")


@dataclass
class PathsConfig:
    data_dir: str = "data"
    db_path: str = "data/netwatch.db"
    models_dir: str = "models"

    def validate(self) -> None:
        for name in ("data_dir", "db_path", "models_dir"):
            _require(bool(getattr(self, name).strip()), f"paths.{name} must not be empty")


@dataclass
class CollectorConfig:
    """how live sockets are sampled (see :mod:`netwatch.collector`)"""

    interval_seconds: float = 5.0
    kind: str = "inet"
    allow_partial: bool = True

    def validate(self) -> None:
        _require(self.interval_seconds > 0, "collector.interval_seconds must be > 0")
        _require(
            self.kind in _PSUTIL_KINDS,
            f"collector.kind must be one of {sorted(_PSUTIL_KINDS)}, got {self.kind!r}",
        )


@dataclass
class FeaturesConfig:
    """how raw events turn into per-process feature windows"""

    window_seconds: int = 300
    min_periodic_events: int = 4
    suspicious_paths: list[str] = field(
        default_factory=lambda: [
            "/tmp",
            "/private/tmp",
            "/var/tmp",
            "/private/var/tmp",
            "/dev/shm",
            "~/Downloads",
        ]
    )

    def validate(self) -> None:
        _require(self.window_seconds >= 30, "features.window_seconds must be >= 30")
        _require(self.min_periodic_events >= 3, "features.min_periodic_events must be >= 3")
        _require(
            all(p.strip() for p in self.suspicious_paths),
            "features.suspicious_paths entries must not be empty",
        )


@dataclass
class NoveltyConfig:
    """time-decayed memory of (process, destination) pairs that are familiar"""

    half_life_days: float = 14.0
    min_familiarity: float = 0.5

    def validate(self) -> None:
        _require(self.half_life_days > 0, "novelty.half_life_days must be > 0")
        # One sighting gives familiarity 1.0, so with a threshold above 1 a single
        # observation could never make a pair known
        _require(
            0 < self.min_familiarity <= 1,
            "novelty.min_familiarity must be within (0, 1]",
        )


@dataclass
class NoveltyDetectorConfig:
    """layer 1 detector settings"""

    enabled: bool = True
    weight: float = 1.0
    unseen_process_surprise: float = 4.0

    def validate(self) -> None:
        _require_weight(self.weight, "detectors.novelty.weight")
        _require(
            self.unseen_process_surprise >= 0,
            "detectors.novelty.unseen_process_surprise must be >= 0",
        )


@dataclass
class BaselineDetectorConfig:
    """layer 1b detector settings"""

    enabled: bool = True
    weight: float = 1.0
    min_windows_per_process: int = 50

    def validate(self) -> None:
        _require_weight(self.weight, "detectors.baseline.weight")
        _require(
            self.min_windows_per_process >= 10,
            "detectors.baseline.min_windows_per_process must be >= 10",
        )


@dataclass
class IsolationForestConfig:
    """layer 2 detector settings"""

    enabled: bool = True
    weight: float = 1.0
    n_estimators: int = 100
    min_windows_per_process: int = 50
    random_state: int = 42

    def validate(self) -> None:
        _require_weight(self.weight, "detectors.isolation_forest.weight")
        _require(self.n_estimators >= 10, "detectors.isolation_forest.n_estimators must be >= 10")
        _require(
            self.min_windows_per_process >= 10,
            "detectors.isolation_forest.min_windows_per_process must be >= 10",
        )


@dataclass
class PlannedDetectorConfig:
    """settings placeholder for a detector layer that is not implemented yet"""

    enabled: bool = False
    weight: float = 1.0


@dataclass
class DetectorsConfig:
    """which detector layers run, and weights for each's opinion"""

    novelty: NoveltyDetectorConfig = field(default_factory=NoveltyDetectorConfig)
    baseline: BaselineDetectorConfig = field(default_factory=BaselineDetectorConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    autoencoder: PlannedDetectorConfig = field(default_factory=PlannedDetectorConfig)
    sequence: PlannedDetectorConfig = field(default_factory=PlannedDetectorConfig)

    def validate(self) -> None:
        self.novelty.validate()
        self.baseline.validate()
        self.isolation_forest.validate()
        # Fail at config time
        for name in ("autoencoder", "sequence"):
            _require_weight(getattr(self, name).weight, f"detectors.{name}.weight")
            _require(
                not getattr(self, name).enabled,
                f"detectors.{name} is a planned detector layer and is not implemented yet. "
                "Set enabled: false",
            )
        _require(
            self.novelty.enabled or self.baseline.enabled or self.isolation_forest.enabled,
            "at least one detector must be enabled",
        )


@dataclass
class TrainingConfig:
    """which windows `netwatch train` learns from"""

    days: int = 7
    warmup_hours: float = 24.0

    def validate(self) -> None:
        _require(self.days >= 1, "training.days must be >= 1")
        _require(self.warmup_hours >= 0, "training.warmup_hours must be >= 0")


@dataclass
class ScoringConfig:
    """how per-detector scores become an alert decision"""

    alert_threshold: float = 0.999
    contribution_threshold: float = 0.95
    min_process_history: int = 50

    def validate(self) -> None:
        _require(0 < self.alert_threshold <= 1, "scoring.alert_threshold must be within (0, 1]")
        _require(
            0 <= self.contribution_threshold <= 1,
            "scoring.contribution_threshold must be within [0, 1]",
        )
        _require(self.min_process_history >= 1, "scoring.min_process_history must be >= 1")


@dataclass
class AlertsConfig:
    """where alerts are delivered to"""

    console: bool = True
    macos_notifications: bool = False

    def validate(self) -> None:
        """Nothing to check: both fields are on/off switches"""


@dataclass
class Config:
    """Complete netwatch configuration.

    ``base_dir`` isn't read from the file. It records where the config file is, so that
    relative paths can be resolved against it
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    detectors: DetectorsConfig = field(default_factory=DetectorsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    base_dir: Path = field(default_factory=Path.cwd, compare=False, repr=False)

    def validate(self) -> Config:
        """validate every section; returns ``self`` so calls can be chained"""
        for section in (
            self.paths,
            self.collector,
            self.features,
            self.novelty,
            self.detectors,
            self.training,
            self.scoring,
            self.alerts,
        ):
            section.validate()
        # A percentile tops out at 1.0, so `weight * percentile` can only reach the alert
        # threshold when the weight does. With no such detector nothing could ever alert
        detectors = self.detectors
        _require(
            any(
                d.enabled and d.weight >= self.scoring.alert_threshold
                for d in (detectors.novelty, detectors.baseline, detectors.isolation_forest)
            ),
            "no enabled detector has a weight of at least scoring.alert_threshold "
            f"({self.scoring.alert_threshold:g}), so nothing could ever raise an alert",
        )
        return self

    def resolve(self, path: str) -> Path:
        """resolve a config path (``~`` and relative paths) to absolute"""
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (self.base_dir / candidate).resolve()

    @property
    def db_path(self) -> Path:
        return self.resolve(self.paths.db_path)

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def models_dir(self) -> Path:
        return self.resolve(self.paths.models_dir)

    @property
    def model_path(self) -> Path:
        return self.models_dir / MODEL_FILENAME


# Generic dict to dataclass conversion


def _build(cls: type[Any], data: Any, where: str) -> Any:
    """build dataclass ``cls`` from YAML mapping"""
    label = where or "config"
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{label} must be a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    loadable = {f.name for f in dataclasses.fields(cls) if f.name != "base_dir"}
    unknown = sorted(set(data) - loadable)
    if unknown:
        raise ConfigError(f"unknown key(s) in {label}: {', '.join(map(str, unknown))}")

    kwargs = {
        name: _coerce(hints[name], value, f"{where}.{name}" if where else name)
        for name, value in data.items()
    }
    return cls(**kwargs)


def _coerce(expected: Any, value: Any, where: str) -> Any:
    """check ``value`` against annotated type, converting int to float"""
    if dataclasses.is_dataclass(expected):
        return _build(expected, value, where)
    if get_origin(expected) is list:
        (item_type,) = get_args(expected)
        if not isinstance(value, list):
            raise ConfigError(f"{where} must be a list")
        return [_coerce(item_type, item, f"{where}[{i}]") for i, item in enumerate(value)]
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    raise ConfigError(f"{where}: unsupported config type {expected!r}")


def config_from_dict(data: dict[str, Any] | None, base_dir: Path | None = None) -> Config:
    """build + validate a :class:`Config` from a parsed mapping"""
    config: Config = _build(Config, data, "")
    config.base_dir = (base_dir or Path.cwd()).resolve()
    return config.validate()


def load_config(path: Path | None = None) -> Config:
    """load config

    Args:
        path: Explicit config file. If ``None``, ``./config.yaml`` is used when
            it exists, otherwise defaults
    """
    if path is None:
        candidate = Path.cwd() / DEFAULT_CONFIG_FILENAME
        if not candidate.is_file():
            return config_from_dict({}, Path.cwd())
        path = candidate

    path = path.expanduser()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    return config_from_dict(data, path.parent)


def example_config_text() -> str:
    """return the text of the example config"""
    packaged = resources.files("netwatch").joinpath(EXAMPLE_CONFIG_FILENAME)
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    repo_copy = Path(__file__).resolve().parents[2] / EXAMPLE_CONFIG_FILENAME
    if repo_copy.is_file():
        return repo_copy.read_text(encoding="utf-8")
    raise ConfigError(f"could not locate {EXAMPLE_CONFIG_FILENAME}")
