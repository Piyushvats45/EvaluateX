"""
database.py
-----------
Zero-cost persistence layer (SQLite — built into Python, no server, no
hosting fees) for logging every prompt, every generated response, and every
computed metric. This is what makes experiments reproducible and lets
report.py analyze results after the fact without re-running any model.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    task TEXT,
    prompt_text TEXT NOT NULL,
    reference_text TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    backend TEXT NOT NULL,
    response_text TEXT NOT NULL,
    latency_sec REAL,
    created_at REAL NOT NULL,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER NOT NULL,
    rouge_l REAL,
    semantic_sim REAL,
    semantic_backend TEXT,
    distinct_2 REAL,
    length INTEGER,
    composite REAL,
    human_score REAL,
    FOREIGN KEY (response_id) REFERENCES responses(id)
);
"""


class ExperimentDB:
    def __init__(self, db_path: str = "outputs/experiments.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- writes ----------------------------------------------------
    def create_experiment(self, name: str, notes: str = "") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO experiments (name, created_at, notes) VALUES (?, ?, ?)",
                (name, time.time(), notes),
            )
            return cur.lastrowid

    def add_prompt(self, experiment_id: int, task: str, prompt_text: str, reference_text: str | None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO prompts (experiment_id, task, prompt_text, reference_text) VALUES (?, ?, ?, ?)",
                (experiment_id, task, prompt_text, reference_text),
            )
            return cur.lastrowid

    def add_response(self, prompt_id: int, model_name: str, backend: str, response_text: str, latency_sec: float) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO responses (prompt_id, model_name, backend, response_text, latency_sec, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (prompt_id, model_name, backend, response_text, latency_sec, time.time()),
            )
            return cur.lastrowid

    def add_metrics(self, response_id: int, metric_result, human_score: float | None = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO metrics (response_id, rouge_l, semantic_sim, semantic_backend, "
                "distinct_2, length, composite, human_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    response_id,
                    metric_result.rouge_l,
                    metric_result.semantic_sim,
                    metric_result.semantic_backend,
                    metric_result.distinct_2,
                    metric_result.length,
                    metric_result.composite,
                    human_score,
                ),
            )
            return cur.lastrowid

    # ---- reads -------------------------------------------------------
    def fetch_results(self, experiment_id: int):
        """Returns a flat list of dict rows joining prompts+responses+metrics,
        ready to be loaded into a pandas DataFrame for reporting."""
        query = """
        SELECT
            p.id AS prompt_id, p.task, p.prompt_text, p.reference_text,
            r.id AS response_id, r.model_name, r.backend, r.response_text, r.latency_sec,
            m.rouge_l, m.semantic_sim, m.semantic_backend, m.distinct_2, m.length,
            m.composite, m.human_score
        FROM prompts p
        JOIN responses r ON r.prompt_id = p.id
        LEFT JOIN metrics m ON m.response_id = r.id
        WHERE p.experiment_id = ?
        ORDER BY p.id, r.model_name
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (experiment_id,)).fetchall()
            return [dict(row) for row in rows]
