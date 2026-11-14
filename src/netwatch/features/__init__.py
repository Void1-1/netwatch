"""Feature extraction: connection events -> per-process feature windows

* :mod:`~netwatch.features.encoders`: small helper functions (time of day, ports, suspicious
  paths, timing regularity)
* :mod:`~netwatch.features.novelty`: which (process, destination) pairs are familiar
* :mod:`~netwatch.features.windowing`: groups events into windows and computes their features
"""
