import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    library TEXT,
    target TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    started_at REAL NOT NULL,
    finished_at REAL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    detail TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS epochs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    function TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    covered_funcs INTEGER,
    covered_edges INTEGER,
    loss REAL,
    metadata TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
    UNIQUE(run_id, function, epoch)
);

CREATE TABLE IF NOT EXISTS corpora (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    function TEXT NOT NULL,
    path TEXT NOT NULL,
    snapshot_path TEXT,
    size_bytes INTEGER,
    created_at REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS corpus_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    function TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    data BLOB NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS policy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function TEXT NOT NULL,
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    value REAL NOT NULL,
    UNIQUE(function, state, action)
);
"""


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self):
        with self.conn:
            yield

    def start_run(self, kind: str, library: Optional[str] = None, target: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        now = time.time()
        meta_json = json.dumps(metadata or {})
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO runs(kind, library, target, started_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (kind, library, target, now, meta_json),
            )
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        meta_json = json.dumps(metadata or {})
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, metadata = ? WHERE id = ?",
                (status, now, meta_json, run_id),
            )

    def record_event(self, run_id: int, label: str, detail: Optional[str] = None) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO events(run_id, label, detail, created_at) VALUES (?, ?, ?, ?)",
                (run_id, label, detail, now),
            )

    def record_epoch(
        self,
        run_id: int,
        function: str,
        epoch: int,
        covered_funcs: int,
        covered_edges: int,
        loss: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO epochs(run_id, function, epoch, covered_funcs, covered_edges, loss, metadata, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, function, epoch) DO UPDATE SET
                    covered_funcs=excluded.covered_funcs,
                    covered_edges=excluded.covered_edges,
                    loss=excluded.loss,
                    metadata=excluded.metadata,
                    created_at=excluded.created_at
                """,
                (run_id, function, epoch, covered_funcs, covered_edges, loss, meta_json, now),
            )

    def record_corpus_snapshot(self, run_id: int, function: str, path: Path, snapshot_path: Optional[Path]) -> None:
        size = path.stat().st_size if path.exists() else None
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO corpora(run_id, function, path, snapshot_path, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, function, str(path), str(snapshot_path) if snapshot_path else None, size, now),
            )

    def record_corpus_blob(self, run_id: int, function: str, epoch: int, blob: bytes) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO corpus_history(run_id, function, epoch, data, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, function, epoch, sqlite3.Binary(blob), now),
            )

    def load_policy(self, function: str) -> Dict[tuple, float]:
        cursor = self.conn.execute("SELECT state, action, value FROM policy WHERE function = ?", (function,))
        return {(row["state"], row["action"]): row["value"] for row in cursor.fetchall()}

    def save_policy(self, function: str, table: Dict[tuple, float]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM policy WHERE function = ?", (function,))
            self.conn.executemany(
                "INSERT INTO policy(function, state, action, value) VALUES (?, ?, ?, ?)",
                ((function, state, action, value) for (state, action), value in table.items()),
            )

    def summarize_runs(self) -> Iterable[sqlite3.Row]:
        cursor = self.conn.execute("SELECT * FROM runs ORDER BY started_at DESC")
        return cursor.fetchall()


def open_storage(db_path: Path) -> Storage:
    return Storage(db_path)
