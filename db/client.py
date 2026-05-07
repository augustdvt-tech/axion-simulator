"""
Axion AI - Database Client
==========================

Thin psycopg2 wrapper for persisting process samples, recommendations, and
operator decisions to TimescaleDB. All public methods are synchronous and
safe to call from FastAPI route handlers (they're fast — bulk inserts, no ORM).

Usage:
    client = DbClient(os.environ["AXION_DB_URL"])
    client.connect()
    client.insert_samples(df, scenario="thermal_drift")
    client.close()

Graceful degradation: if the DB is unavailable the server should catch the
exception at connect time and set db=None. Every downstream caller guards on
`if state.db` so no writes are attempted when the DB is offline.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import execute_values

# Mapping: CSV/DataFrame dotted tag  →  DB column name (underscores)
_TAG_TO_COL = {
    "cstr.T_R_C":       "cstr_T_R_C",
    "cstr.T_J_C":       "cstr_T_J_C",
    "cstr.C_A":         "cstr_C_A",
    "cstr.F_feed":      "cstr_F_feed",
    "cstr.F_cool":      "cstr_F_cool",
    "cstr.T_feed_C":    "cstr_T_feed_C",
    "cstr.T_cool_in_C": "cstr_T_cool_in_C",
    "cstr.P_R":         "cstr_P_R",
    "cstr.conversion":  "cstr_conversion",
    "column.x_D":       "column_x_D",
    "column.x_B_A":     "column_x_B_A",
    "column.purity_B":  "column_purity_B",
    "column.T_top_C":   "column_T_top_C",
    "column.T_bot_C":   "column_T_bot_C",
    "column.RR":        "column_RR",
    "column.F_vap_kgh": "column_F_vap_kgh",
    "column.Q_reb_kW":  "column_Q_reb_kW",
    "column.P_top_bar": "column_P_top_bar",
    "column.P_bot_bar": "column_P_bot_bar",
}

_SAMPLE_COLS = ["timestamp", "scenario"] + list(_TAG_TO_COL.values())
_COL_TO_TAG  = {v: k for k, v in _TAG_TO_COL.items()}   # DB col → dotted tag
_BATCH_SIZE  = 1000
_MAX_LIMIT   = 10_000


def _serialize(value: Any) -> Any:
    """Convert DB-native types to JSON-safe primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class DbClient:
    def __init__(self, db_url: str) -> None:
        self._url  = db_url
        self._conn: Optional[psycopg2.extensions.connection] = None

    # ------------------------------------------------------------------ #
    # Connection lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        self._conn = psycopg2.connect(self._url)
        self._conn.autocommit = False

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
        self._conn = None

    # ------------------------------------------------------------------ #
    # Scenarios                                                            #
    # ------------------------------------------------------------------ #

    def upsert_scenario(self, name: str, n_samples: int) -> None:
        sql = """
            INSERT INTO scenarios (name, n_samples, ingested_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (name) DO UPDATE
              SET n_samples   = EXCLUDED.n_samples,
                  ingested_at = EXCLUDED.ingested_at
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (name, n_samples))
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Process samples                                                      #
    # ------------------------------------------------------------------ #

    def insert_samples(self, df, scenario: str) -> None:
        """Bulk-insert all rows of a scenario DataFrame.

        Idempotent: ON CONFLICT DO NOTHING on (timestamp, scenario).
        Dots in column names are mapped to underscores to match the schema.
        """
        rows: List[tuple] = []
        for _, row in df.iterrows():
            values = [row["timestamp"], scenario] + [
                float(row[tag]) if tag in row.index else None
                for tag in _TAG_TO_COL
            ]
            rows.append(tuple(values))

        col_str = ", ".join(_SAMPLE_COLS)
        sql = f"""
            INSERT INTO process_samples ({col_str})
            VALUES %s
            ON CONFLICT (timestamp, scenario) DO NOTHING
        """
        with self._conn.cursor() as cur:
            for i in range(0, len(rows), _BATCH_SIZE):
                execute_values(cur, sql, rows[i : i + _BATCH_SIZE])
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def upsert_recommendations(self, recs, scenario: str) -> None:
        """Insert recommendations generated for a scenario run.

        Uses ON CONFLICT (id) DO UPDATE so re-loading the same scenario
        refreshes statuses rather than duplicating rows.
        """
        if not recs:
            return
        sql = """
            INSERT INTO recommendations
              (id, scenario, timestamp, rule_id, urgency, confidence,
               diagnosis, action, status)
            VALUES %s
            ON CONFLICT (id) DO UPDATE
              SET status = EXCLUDED.status
        """
        rows = [
            (
                r.id,
                scenario,
                r.timestamp,
                r.rule_fired,
                r.urgency.value,
                r.confidence,
                r.diagnosis,
                r.action.description if r.action else None,
                r.status,
            )
            for r in recs
        ]
        with self._conn.cursor() as cur:
            execute_values(cur, sql, rows)
        self._conn.commit()

    def update_recommendation_status(self, rec_id: str, status: str) -> None:
        sql = "UPDATE recommendations SET status = %s WHERE id = %s"
        with self._conn.cursor() as cur:
            cur.execute(sql, (status, rec_id))
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Decisions                                                            #
    # ------------------------------------------------------------------ #

    def insert_decision(self, rec_id: str, decision: str, rationale: str) -> None:
        """Record a single operator decision (accept / modify / reject)."""
        sql = """
            INSERT INTO decisions (recommendation_id, decision, rationale)
            VALUES (%s, %s, %s)
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (rec_id, decision, rationale))
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Read / query methods                                                 #
    # ------------------------------------------------------------------ #

    def list_ingested_scenarios(self) -> List[Dict[str, Any]]:
        """Return all rows from the scenarios table, newest first."""
        sql = "SELECT name, n_samples, ingested_at FROM scenarios ORDER BY ingested_at DESC"
        with self._conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            return [
                {k: _serialize(v) for k, v in zip(cols, row)}
                for row in cur.fetchall()
            ]

    def query_samples(
        self,
        scenario: Optional[str] = None,
        from_ts:  Optional[str] = None,
        to_ts:    Optional[str] = None,
        tags:     Optional[List[str]] = None,
        limit:    int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query process_samples. Returns rows with dotted-tag keys.

        tags: list of dotted names (e.g. ["cstr.T_R_C", "column.purity_B"]).
              If None, all sensor columns are returned.
        """
        limit = min(limit, _MAX_LIMIT)

        # Build SELECT column list
        if tags:
            sensor_cols = [_TAG_TO_COL[t] for t in tags if t in _TAG_TO_COL]
        else:
            sensor_cols = list(_TAG_TO_COL.values())
        select_cols = ["timestamp", "scenario"] + sensor_cols

        # Build WHERE
        conditions: List[str] = []
        params: List[Any] = []
        if scenario:
            conditions.append("scenario = %s")
            params.append(scenario)
        if from_ts:
            conditions.append("timestamp >= %s")
            params.append(from_ts)
        if to_ts:
            conditions.append("timestamp <= %s")
            params.append(to_ts)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        col_str = ", ".join(select_cols)
        sql = (f"SELECT {col_str} FROM process_samples "
               f"{where} ORDER BY timestamp ASC LIMIT %s")
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            db_cols = [d[0] for d in cur.description]
            rows = cur.fetchall()

        # Remap underscore DB names back to dotted tags
        result = []
        for row in rows:
            result.append({
                _COL_TO_TAG.get(k, k): _serialize(v)
                for k, v in zip(db_cols, row)
            })
        return result

    def query_recommendations(
        self,
        scenario: Optional[str] = None,
        from_ts:  Optional[str] = None,
        to_ts:    Optional[str] = None,
        status:   Optional[List[str]] = None,
        urgency:  Optional[List[str]] = None,
        limit:    int = 200,
    ) -> List[Dict[str, Any]]:
        """Query recommendations table with optional filters."""
        limit = min(limit, _MAX_LIMIT)

        conditions: List[str] = []
        params: List[Any] = []
        if scenario:
            conditions.append("scenario = %s")
            params.append(scenario)
        if from_ts:
            conditions.append("timestamp >= %s")
            params.append(from_ts)
        if to_ts:
            conditions.append("timestamp <= %s")
            params.append(to_ts)
        if status:
            placeholders = ", ".join(["%s"] * len(status))
            conditions.append(f"status IN ({placeholders})")
            params.extend(status)
        if urgency:
            placeholders = ", ".join(["%s"] * len(urgency))
            conditions.append(f"urgency IN ({placeholders})")
            params.extend(urgency)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (f"SELECT id, scenario, timestamp, rule_id, urgency, confidence, "
               f"diagnosis, action, status, created_at "
               f"FROM recommendations {where} ORDER BY timestamp DESC LIMIT %s")
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [
                {k: _serialize(v) for k, v in zip(cols, row)}
                for row in cur.fetchall()
            ]

    def query_decisions(
        self,
        scenario: Optional[str] = None,
        from_ts:  Optional[str] = None,
        to_ts:    Optional[str] = None,
        limit:    int = 200,
    ) -> List[Dict[str, Any]]:
        """Query decisions joined with recommendations for scenario context."""
        limit = min(limit, _MAX_LIMIT)

        conditions: List[str] = []
        params: List[Any] = []
        if scenario:
            conditions.append("r.scenario = %s")
            params.append(scenario)
        if from_ts:
            conditions.append("d.decided_at >= %s")
            params.append(from_ts)
        if to_ts:
            conditions.append("d.decided_at <= %s")
            params.append(to_ts)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (f"SELECT d.id, d.recommendation_id, d.decision, d.rationale, "
               f"d.decided_at, r.scenario, r.urgency, r.rule_id, r.diagnosis "
               f"FROM decisions d "
               f"JOIN recommendations r ON d.recommendation_id = r.id "
               f"{where} ORDER BY d.decided_at DESC LIMIT %s")
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [
                {k: _serialize(v) for k, v in zip(cols, row)}
                for row in cur.fetchall()
            ]
