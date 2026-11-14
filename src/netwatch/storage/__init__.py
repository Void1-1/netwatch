"""Storage: one local SQLite database holds all netwatch state

See :mod:`netwatch.storage.db` for the access layer and ``schema.sql`` for the tables. Nothing in
this package uses the network
"""

from netwatch.storage.db import Database

__all__ = ["Database"]
