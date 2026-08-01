"""db.py — SQLite persistence for the outreach tracker.

Stores every qualified company + your outreach status in tracker.db, deduped on
(profile, company_number). Re-running a batch adds new companies and leaves
existing rows untouched.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("tracker.db")
STATUSES = ["New", "Connection sent", "Accepted", "Messaged", "Replied", "Skip"]

_ENGINE_FIELDS = [
    "company", "ch_company_name", "linkedin_name", "website", "target_personas",
    "sector", "town", "accounts_type", "google_search", "linkedin_company_search",
    "ch_url", "company_number", "sic_codes", "connection_note", "followup_dm",
]
_USER_FIELDS = ["status", "contact_name", "contact_url", "user_notes"]


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile TEXT NOT NULL,
                company_number TEXT NOT NULL,
                company TEXT, ch_company_name TEXT, linkedin_name TEXT, website TEXT,
                target_personas TEXT, sector TEXT, town TEXT, accounts_type TEXT,
                google_search TEXT, linkedin_company_search TEXT, ch_url TEXT,
                sic_codes TEXT, connection_note TEXT, followup_dm TEXT,
                status TEXT DEFAULT 'New',
                contact_name TEXT DEFAULT '', contact_url TEXT DEFAULT '',
                user_notes TEXT DEFAULT '',
                created_at TEXT, updated_at TEXT,
                UNIQUE(profile, company_number)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                profile TEXT PRIMARY KEY, next_skip INTEGER DEFAULT 0
            )
        """)


def upsert_lead(profile: str, d: dict[str, Any]) -> bool:
    number = d.get("company_number") or ""
    if not number:
        return False
    cols = ["profile"] + _ENGINE_FIELDS
    vals = [profile] + [d.get(f, "") for f in _ENGINE_FIELDS]
    placeholders = ", ".join("?" for _ in cols)
    with _conn() as c:
        cur = c.execute(
            f"INSERT OR IGNORE INTO leads ({', '.join(cols)}, created_at, updated_at) "
            f"VALUES ({placeholders}, ?, ?)", (*vals, _now(), _now()))
        return cur.rowcount > 0


def fetch_leads(profile: Optional[str] = None, status: Optional[str] = None,
                sector: Optional[str] = None) -> list[dict]:
    q, params = "SELECT * FROM leads WHERE 1=1", []
    if profile:
        q += " AND profile = ?"; params.append(profile)
    if status and status != "All":
        q += " AND status = ?"; params.append(status)
    if sector and sector != "All":
        q += " AND sector = ?"; params.append(sector)
    q += " ORDER BY sector, ch_company_name"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, params).fetchall()]


def update_lead(lead_id: int, fields: dict[str, Any]) -> None:
    fields = {k: v for k, v in fields.items() if k in _USER_FIELDS}
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE leads SET {sets}, updated_at = ? WHERE id = ?",
                  (*fields.values(), _now(), lead_id))


def status_counts(profile: str) -> dict[str, int]:
    with _conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM leads WHERE profile = ? GROUP BY status",
                         (profile,)).fetchall()
    return {r["status"]: r["n"] for r in rows}


def sectors(profile: str) -> list[str]:
    with _conn() as c:
        rows = c.execute("SELECT DISTINCT sector FROM leads WHERE profile = ? ORDER BY sector",
                         (profile,)).fetchall()
    return [r["sector"] for r in rows if r["sector"]]


def get_next_skip(profile: str) -> int:
    with _conn() as c:
        r = c.execute("SELECT next_skip FROM progress WHERE profile = ?", (profile,)).fetchone()
    return r["next_skip"] if r else 0


def set_next_skip(profile: str, n: int) -> None:
    with _conn() as c:
        c.execute("INSERT INTO progress (profile, next_skip) VALUES (?, ?) "
                  "ON CONFLICT(profile) DO UPDATE SET next_skip = ?", (profile, n, n))
