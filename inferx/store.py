"""On-device memory store: SQLite plus sqlite-vec, split into three tiers.

core:     short stable facts that are always placed in context
recall:   every raw message
archival: standalone facts written by the controller

Replaced facts are marked as superseded, never erased.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import DIM
from .interface import Message

VEC_EXTENSION = Path(__file__).resolve().parent.parent / "third_party" / "sqlite-vec" / "vec0.dll"

SCHEMA = f"""
create table core(key text primary key, value text not null, source text);
create table recall(
  rowid integer primary key, id text unique, speaker text, text text, date text,
  session integer, rendered text);
create table archival(
  rowid integer primary key, text text not null, source text, date text,
  superseded_by integer references archival(rowid));
create table queue(rowid integer primary key, message_rowid integer not null);
create table outdated(id text primary key, replaced_by text);
create virtual table recall_vec using vec0(e float[{DIM}] distance_metric=cosine);
create virtual table archival_vec using vec0(e float[{DIM}] distance_metric=cosine);
"""


@dataclass(frozen=True)
class Hit:
    tier: str
    rowid: int
    text: str
    source: str
    score: float


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.enable_load_extension(True)
        self.db.load_extension(str(VEC_EXTENSION))
        self.db.enable_load_extension(False)
        self.db.executescript(SCHEMA)

    def add_message(self, message: Message, vector: np.ndarray) -> int:
        with self.db:
            rowid = self.db.execute(
                "insert into recall(id, speaker, text, date, session, rendered) values (?,?,?,?,?,?)",
                (message.id, message.speaker, message.text, message.date, message.session, message.rendered),
            ).lastrowid
            self.db.execute(
                "insert into recall_vec(rowid, e) values (?, ?)", (rowid, vector.tobytes())
            )
        return rowid

    def enqueue(self, message_rowid: int) -> None:
        with self.db:
            self.db.execute("insert into queue(message_rowid) values (?)", (message_rowid,))

    def queued(self, limit: int | None = None) -> list[tuple[int, Message]]:
        sql = """select q.rowid, r.id, r.speaker, r.text, r.date, r.session
                 from queue q join recall r on r.rowid = q.message_rowid order by q.rowid"""
        rows = self.db.execute(sql + (" limit ?" if limit else ""), (limit,) if limit else ()).fetchall()
        return [(row[0], Message(row[1], row[2], row[3], row[4], row[5])) for row in rows]

    def dequeue(self, queue_rowid: int) -> None:
        with self.db:
            self.db.execute("delete from queue where rowid = ?", (queue_rowid,))

    def add_fact(self, text: str, source: str, date: str, vector: np.ndarray) -> int:
        with self.db:
            rowid = self.db.execute(
                "insert into archival(text, source, date) values (?,?,?)", (text, source, date)
            ).lastrowid
            self.db.execute(
                "insert into archival_vec(rowid, e) values (?, ?)", (rowid, vector.tobytes())
            )
        return rowid

    def supersede(self, old_rowid: int, new_rowid: int) -> None:
        with self.db:
            self.db.execute(
                "update archival set superseded_by = ? where rowid = ?", (new_rowid, old_rowid)
            )

    def retire_facts(self, source: str, new_rowid: int | None) -> None:
        """Hide the facts drawn from a replaced message. 0 means its replacement has no fact."""
        with self.db:
            self.db.execute("update archival set superseded_by = ? where source = ? and superseded_by is null",
                            (new_rowid or 0, source))

    def mark_outdated(self, message_id: str, replaced_by: str) -> None:
        with self.db:
            self.db.execute("insert or ignore into outdated(id, replaced_by) values (?, ?)", (message_id, replaced_by))

    def message(self, message_id: str) -> Message | None:
        row = self.db.execute("select id, speaker, text, date, session from recall where id = ?", (message_id,)).fetchone()
        return Message(*row) if row else None

    def has_fact(self, message_id: str) -> bool:
        return self.db.execute("select 1 from archival where source = ? and superseded_by is null",
                               (message_id,)).fetchone() is not None

    def earlier_versions(self, message_id: str) -> list[Hit]:
        """Walk back through what this message replaced, newest first."""
        hits = []
        current = message_id
        while True:
            row = self.db.execute(
                """select r.rowid, r.rendered, r.id from outdated o join recall r on r.id = o.id
                   where o.replaced_by = ?""", (current,)).fetchone()
            if row is None:
                return hits
            hits.append(Hit("recall", row[0], row[1], row[2], 0.0))
            current = row[2]

    def outdated(self) -> set[str]:
        return {row[0] for row in self.db.execute("select id from outdated")}

    def remove_core(self, key: str) -> None:
        with self.db:
            self.db.execute("delete from core where key = ?", (key,))

    def set_core(self, key: str, value: str, source: str) -> None:
        with self.db:
            self.db.execute(
                "insert into core(key, value, source) values (?,?,?) "
                "on conflict(key) do update set value = excluded.value, source = excluded.source",
                (key, value, source),
            )

    def core(self) -> list[str]:
        return [row[0] for row in self.db.execute("select value from core order by key")]

    def counts(self) -> dict:
        one = lambda sql: self.db.execute(sql).fetchone()[0]
        return {
            "recall": one("select count(*) from recall"),
            "facts_active": one("select count(*) from archival where superseded_by is null"),
            "facts_superseded": one("select count(*) from archival where superseded_by is not null"),
            "core": one("select count(*) from core"),
            "queued": one("select count(*) from queue"),
            "outdated": one("select count(*) from outdated"),
        }

    def search_recall(self, vector: np.ndarray, k: int) -> list[Hit]:
        rows = self.db.execute(
            """select r.rowid, r.rendered, r.id, v.distance from
               (select rowid, distance from recall_vec where e match ? and k = ?) v
               join recall r on r.rowid = v.rowid order by v.distance""",
            (vector.tobytes(), k),
        ).fetchall()
        return [Hit("recall", row[0], row[1], row[2], 1 - row[3]) for row in rows]

    def search_facts(self, vector: np.ndarray, k: int) -> list[Hit]:
        """Return active facts only. Over-fetches because vec0 can't filter on superseded."""
        rows = self.db.execute(
            """select a.rowid, a.text, a.source, a.date, a.superseded_by, v.distance from
               (select rowid, distance from archival_vec where e match ? and k = ?) v
               join archival a on a.rowid = v.rowid order by v.distance""",
            (vector.tobytes(), k * 3),
        ).fetchall()
        hits = []
        for rowid, text, source, date, superseded_by, distance in rows:
            if superseded_by is None:
                rendered = f"[{date}] {text}" if date else text
                hits.append(Hit("archival", rowid, rendered, source, 1 - distance))
        return hits[:k]
