"""
Axion AI - FastAPI Server
=========================

REST + WebSocket interface to the Axion AI pipeline.

Architecture
------------
On startup, the server:
    1. Loads the training scenario (`normal.csv`) and fits the analytical engine.
    2. Loads the active simulation scenario (configurable; default: thermal_drift).
    3. Pre-computes all sessions, recommendations, decisions, executions and
       outcomes for the entire run. These constitute the canonical timeline.
    4. Exposes the timeline through REST endpoints AND streams live process
       data + alerts through a WebSocket as if happening in real time.

Real-time replay
----------------
The simulation csv is replayed at a configurable speed (default 60x — 1 hour
of simulated time = 1 minute of wall-clock). The WebSocket pushes a fresh
sample to all connected clients on every tick. As the wall-clock crosses
the timestamp of any pre-computed recommendation/decision/outcome, those
events are also pushed, so the UI sees them appear "as they happen".

This is exactly the behavior an operator would experience with live data,
but driven by the deterministic simulator instead of OPC-UA.

Endpoints
---------
GET   /api/health                  - liveness check
GET   /api/scenarios               - list available scenarios
POST  /api/scenarios/select        - switch active scenario, restart replay
GET   /api/state                   - current process snapshot
GET   /api/process/recent          - last N samples (for charts)
GET   /api/recommendations         - all recommendations issued so far
GET   /api/recommendations/{id}    - full detail of one recommendation
POST  /api/recommendations/{id}/decide - record an operator decision
GET   /api/decisions               - decision log
GET   /api/performance             - per-rule performance summary
GET   /api/replay/status           - replay clock and progress

WebSocket
---------
WS   /ws/stream                    - bidirectional stream of all live events
"""

from __future__ import annotations
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analytics import AnalyticalEngine
from analytics.drift import DriftDetector
from recommendations import RecommendationEngine, recommendations_to_dataframe
from consensus import (
    ConsensusController, OperatingMode, RealisticOperator, PerformanceTracker,
    decisions_to_dataframe, outcomes_to_dataframe, DecisionStatus,
)
from consensus.operator_outcomes import (
    OperatorOverride, measure_pending, outcome_summary_dict,
)
from soft_sensor import SoftSensor, SoftSensorDetector, PILOT_PURITY_FEATURES
from optimizer import (
    ProcessSurrogate, NSGA2Optimizer,
    PurityObjective, EnergyObjective, ProductionObjective, StabilityObjective,
)
try:
    from predictive import LSTMForecaster, LSTMPredictiveDetector
except ImportError:
    LSTMForecaster = None          # TensorFlow not available
    LSTMPredictiveDetector = None
from axion_logging import get_logger
from db.client import DbClient
from api.webhooks import WebhookNotifier
from api.auth import (
    AuthError, InvalidTokenError, TokenExpiredError, TokenTypeMismatchError,
    access_ttl_seconds, decode_token, extract_bearer_token,
    hash_password, issue_token_pair, jwt_secret, verify_password,
)
from db.users import UserRepository
from api.metrics import (
    RequestTimer, http_requests_total, operator_decisions_total,
    rate_limit_rejections_total, recommendations_total, render_prometheus,
    template_path, websocket_connections,
)
from api.rate_limit import EXEMPT_PATHS as RL_EXEMPT, RateLimiter, resolve_identity
from api.data_source import OpcuaBuffer, LIVE_COLUMNS
import profile as profile_module
from profile import active_profile, get_profile, list_profiles
from integration.integration_service import IntegrationService

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR   = PROJECT_ROOT / "data"
UI_DIR     = PROJECT_ROOT / "ui"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

REPLAY_SPEED_DEFAULT = 60.0     # 60× real-time
REPLAY_TICK_SECONDS = 1.0       # advance every 1 second wall-clock


# =============================================================================
# State container
# =============================================================================

@dataclass
class ScenarioRun:
    """All pre-computed data for a single scenario run."""
    scenario: str
    process_data: pd.DataFrame              # full simulation data
    sessions: list                          # all event sessions
    recommendations: list                   # all recommendations
    decisions: list                         # all decisions (from sim operator)
    executions: list                        # all executions
    outcomes: list                          # all outcomes
    performance: PerformanceTracker         # per-rule track record
    rec_df: pd.DataFrame                    # recommendations as DataFrame
    dec_df: pd.DataFrame                    # decisions as DataFrame


class AppState:
    """Global server state. Single-active-scenario for the MVP."""
    def __init__(self):
        self.ae: Optional[AnalyticalEngine] = None
        self.re: Optional[RecommendationEngine] = None
        self.run: Optional[ScenarioRun] = None
        self.soft_sensor: Optional[SoftSensor] = None
        self.surrogate: Optional[ProcessSurrogate] = None
        self.forecaster: Optional[LSTMForecaster] = None
        # Cache the most recent Pareto front (recomputed when context changes)
        self.pareto_cache: Optional[dict] = None
        # Replay clock
        self.replay_speed: float = REPLAY_SPEED_DEFAULT
        self.replay_running: bool = True
        self.replay_idx: int = 0
        self.operator_overrides: Dict[str, dict] = {}
        # Outcomes measured for UI-driven decisions, keyed by rec_id.
        self.operator_outcomes: Dict[str, dict] = {}
        self.clients: Set[WebSocket] = set()
        self.db: Optional[DbClient] = None
        self.webhook: WebhookNotifier = WebhookNotifier.from_env()
        self.drift_detector: Optional[DriftDetector] = None
        self.opcua: Optional[IntegrationService] = None
        # Live OPC-UA buffer; replay loop reads from this when data_source == "opcua".
        self.opcua_buffer: OpcuaBuffer = OpcuaBuffer()
        # "replay" (default) or "opcua". Switched via /api/data-source/select.
        self.data_source: str = "replay"
        # Rate limiter — built from env on first use; replaceable in tests.
        self.rate_limiter: RateLimiter = RateLimiter.from_env()


state = AppState()


# =============================================================================
# Scenario loading
# =============================================================================

def list_scenarios() -> List[str]:
    """Scan data directory for available scenarios."""
    return sorted(p.stem for p in DATA_DIR.glob("*.csv"))


def _baseline_csv_for(prof) -> Path:
    """Pick the baseline scenario CSV the AnalyticalEngine fits on.

    Heuristic: prefer the scenario named "normal" (pilot convention) or
    "<profile>_normal" (batch convention). Falls back to the first
    scenario the profile declares.
    """
    candidates = [
        DATA_DIR / "normal.csv",
        DATA_DIR / f"{prof.name}_normal.csv",
    ]
    for s in prof.scenarios:
        candidates.append(DATA_DIR / f"{s}.csv")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No baseline CSV found for profile {prof.name!r}. "
        f"Expected one of: {[str(c) for c in candidates]}"
    )


def load_scenario(scenario: str) -> ScenarioRun:
    """Run the full pipeline on a scenario and return all artifacts."""
    csv_path = DATA_DIR / f"{scenario}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Scenario '{scenario}' not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    sessions = state.ae.run_sessions(df)
    recs = state.re.generate(sessions, df)

    cc = ConsensusController(
        mode=OperatingMode.SEMI_AUTONOMOUS,
        operator=RealisticOperator(seed=42),
    )
    log = cc.process(recs, df)

    rec_df = recommendations_to_dataframe(recs)
    dec_df = decisions_to_dataframe(log.decisions)

    # Persist to DB if available (graceful — never crashes the server)
    if state.db:
        try:
            state.db.upsert_scenario(scenario, len(df))
            state.db.insert_samples(df, scenario)
            state.db.upsert_recommendations(recs, scenario)
            logger.info("Scenario persisted to DB", extra={
                "scenario": scenario,
                "samples": len(df),
                "recommendations": len(recs),
            })
        except Exception as e:
            logger.error("DB persist failed — continuing in memory-only mode",
                         extra={"error": str(e)})

    return ScenarioRun(
        scenario=scenario,
        process_data=df,
        sessions=sessions,
        recommendations=recs,
        decisions=log.decisions,
        executions=log.executions,
        outcomes=log.outcomes,
        performance=cc.performance_tracker,
        rec_df=rec_df,
        dec_df=dec_df,
    )


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="Axion AI", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Observability + rate-limit middleware (Bloque W)
# =============================================================================
#
# Registered BEFORE the auth middleware so at runtime it executes AFTER
# auth (FastAPI processes middlewares in LIFO order). That lets us read
# `request.state.role` / `request.state.user_id` and label the rate
# limiter by user when available.

@app.middleware("http")
async def metrics_and_rate_limit(request: Request, call_next):
    method = request.method
    raw_path = request.url.path
    path_t = template_path(raw_path)

    # Rate limiting — exempt health, metrics, and non-API paths
    if (raw_path.startswith(("/api/", "/ws/"))
            and raw_path not in RL_EXEMPT
            and state.rate_limiter is not None
            and state.rate_limiter.enabled):
        identity = resolve_identity(
            user_id=getattr(request.state, "user_id", None),
            api_key=request.headers.get("X-API-Key", "") or None,
            client_ip=request.client.host if request.client else None,
        )
        allowed, retry_after = state.rate_limiter.allow(identity)
        if not allowed:
            limiter_kind = identity.split(":", 1)[0] if ":" in identity else "anonymous"
            rate_limit_rejections_total.labels(limiter=limiter_kind).inc()
            return JSONResponse(
                status_code=429,
                content={
                    "detail":      "Rate limit exceeded.",
                    "retry_after": round(retry_after, 2),
                },
                headers={"Retry-After": str(int(round(retry_after)) or 1)},
            )

    # Time the request and record the count + status
    with RequestTimer(method=method, path_template=path_t):
        response = await call_next(request)
    http_requests_total.labels(
        method=method, path=path_t, status=str(response.status_code),
    ).inc()
    return response


# =============================================================================
# Auth middleware (RBAC: viewer / operator / manager)
# =============================================================================

# Roles, in order of increasing privilege:
#   viewer    — read-only (GET /api/*)
#   operator  — viewer + record decisions, control replay, run what-if predictions
#   manager   — operator + change scenarios and any other write op
#
# Configuration via environment:
#   AXION_API_KEY              — legacy, full-access (treated as manager)
#   AXION_API_KEY_VIEWER       — grants viewer role
#   AXION_API_KEY_OPERATOR     — grants operator role
#   AXION_API_KEY_MANAGER      — grants manager role
#
# If NO key is set, auth is disabled (dev/smoke-test mode).
#
# Protected paths: /api/* (except /api/health) and /ws/*
# Clients: set X-API-Key header. WebSocket clients: pass ?api_key= query param.

_ROLE_LEVEL = {"viewer": 0, "operator": 1, "manager": 2}


def _load_role_keys() -> Dict[str, str]:
    """Read env vars and build {key → role} mapping. Empty dict ⇒ RBAC off."""
    keys: Dict[str, str] = {}
    legacy = os.environ.get("AXION_API_KEY", "").strip()
    if legacy:
        keys[legacy] = "manager"
    for role in ("viewer", "operator", "manager"):
        v = os.environ.get(f"AXION_API_KEY_{role.upper()}", "").strip()
        if v:
            keys[v] = role
    return keys


def _required_role(method: str, path: str) -> str:
    """Minimum role needed for a given (method, path)."""
    if method == "GET":
        return "viewer"
    if method == "POST":
        # Operator-level write endpoints
        if path.startswith("/api/recommendations/") and path.endswith("/decide"):
            return "operator"
        if path == "/api/replay/control":
            return "operator"
        if path == "/api/optimization/predict":
            return "operator"
    # Everything else (scenario change, future admin ops) requires manager
    return "manager"


def _role_satisfies(actual: str, required: str) -> bool:
    return _ROLE_LEVEL.get(actual, -1) >= _ROLE_LEVEL.get(required, 99)


_AUTH_PUBLIC_PATHS = {
    "/api/health",
    "/api/metrics",
    "/api/auth/login", "/api/auth/refresh",
}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Auth pipeline:
       1. If a `Authorization: Bearer <jwt>` header is present, validate it
          and use its `role` claim.
       2. Otherwise, fall back to the X-API-Key path (Bloque N).
       3. If neither auth backend is configured, allow the request through.
    """
    secret    = jwt_secret()
    role_keys = _load_role_keys()

    path = request.url.path
    method = request.method

    # Always-public paths
    if not path.startswith(("/api/", "/ws/")) or path in _AUTH_PUBLIC_PATHS:
        return await call_next(request)

    # Auth disabled when neither backend is configured
    if secret is None and not role_keys:
        return await call_next(request)

    role: Optional[str] = None
    user_email: Optional[str] = None

    # ---- 1. JWT bearer token ----
    bearer = extract_bearer_token(request.headers.get("Authorization"))
    if secret is not None and bearer:
        try:
            claims = decode_token(bearer, secret, expected_type="access")
            role = claims.role
            user_email = claims.sub
            request.state.user_id = claims.uid
        except TokenExpiredError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Access token expired.", "code": "token_expired"},
            )
        except (InvalidTokenError, TokenTypeMismatchError, AuthError) as e:
            return JSONResponse(
                status_code=401,
                content={"detail": f"Invalid token: {e}", "code": "invalid_token"},
            )

    # ---- 2. API-key fallback ----
    if role is None and role_keys:
        provided = (
            request.headers.get("X-API-Key", "")
            or request.query_params.get("api_key", "")
        )
        role = role_keys.get(provided)
        if role is None:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid or missing credentials.",
                    "hint":   "Provide Authorization: Bearer <jwt>, "
                              "or set the X-API-Key header.",
                },
            )

    # If neither backend produced a role, request is unauthenticated
    if role is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required."},
        )

    required = _required_role(method, path)
    if not _role_satisfies(role, required):
        return JSONResponse(
            status_code=403,
            content={
                "detail":         f"Role '{role}' cannot {method} {path}.",
                "required_role":  required,
                "actual_role":    role,
            },
        )

    request.state.role = role
    if user_email:
        request.state.user_email = user_email
    return await call_next(request)


@app.on_event("shutdown")
async def on_shutdown():
    if state.db:
        state.db.close()
        logger.info("Database connection closed")
    if state.opcua is not None:
        await state.opcua.stop()
        logger.info("OPC-UA integration stopped")


@app.on_event("startup")
async def on_startup():
    logger.info("Axion AI server starting")

    # Optional DB connection — server works without it
    db_url = os.environ.get("AXION_DB_URL")
    if db_url:
        try:
            state.db = DbClient(db_url)
            state.db.connect()
            logger.info("Database connected", extra={"url": db_url.split("@")[-1]})
        except Exception as e:
            state.db = None
            logger.warning("Database unavailable — running in memory-only mode",
                           extra={"error": str(e)})
    # Profile-aware analytical + recommendation engine setup
    profile = active_profile()
    baseline_csv = _baseline_csv_for(profile)
    df_train = pd.read_csv(baseline_csv)
    state.ae = AnalyticalEngine(
        tags=profile.tag_names,
        operational_limits=profile.operational_limits or None,
        training_fraction=1.0,
        warmup_minutes=15.0,
    )
    state.ae.fit(df_train)
    state.re = RecommendationEngine(
        rules=profile.load_rules() or None,
        operational_limits=profile.operational_limits or None,
    )
    logger.info("Analytical engine ready", extra={
        "profile":        profile.name,
        "n_tags":         len(profile.tag_names),
        "pca_components": state.ae.pca.model.n_components if state.ae.pca else 0,
        "n_rules":        len(state.re.rules),
    })

    # Fit the drift detector on the active profile's feature columns
    drift_features = profile.feature_cols or PILOT_PURITY_FEATURES
    state.drift_detector = DriftDetector(features=drift_features).fit(df_train)
    logger.info("Drift detector ready", extra={
        "features":    len(drift_features),
        "ref_samples": len(df_train),
    })

    # Load the pre-trained soft sensor if available
    ss_path = PROJECT_ROOT / "results" / "models" / "purity_soft_sensor.joblib"
    if ss_path.exists():
        try:
            state.soft_sensor = SoftSensor.load(ss_path)
            logger.info("Soft sensor loaded", extra={
                "model": ss_path.name,
                "features": len(state.soft_sensor.feature_names),
            })
        except Exception as e:
            logger.error("Soft sensor load failed", extra={"error": str(e)})
    else:
        logger.warning("Soft sensor model not found", extra={
            "path": str(ss_path),
            "hint": "run: python examples/train_soft_sensor.py",
        })

    # Load the process surrogate (analytical model — fast to instantiate)
    state.surrogate = ProcessSurrogate()
    logger.info("Process surrogate ready", extra={"inputs": len(state.surrogate.inputs)})

    # Load the LSTM forecaster (Tarea 7) if available
    lstm_dir = PROJECT_ROOT / "results" / "models" / "lstm_forecaster"
    if lstm_dir.exists() and (lstm_dir / "model.keras").exists():
        try:
            state.forecaster = LSTMForecaster.load(lstm_dir)
            logger.info("LSTM forecaster loaded", extra={
                "targets": len(state.forecaster.target_cols),
                "horizons_min": state.forecaster.config.horizons_minutes,
            })
        except Exception as e:
            logger.error("LSTM forecaster load failed", extra={"error": str(e)})
    else:
        logger.warning("LSTM forecaster not found", extra={
            "path": str(lstm_dir),
            "hint": "run: python examples/train_lstm_forecaster.py",
        })

    # Default scenario
    default_scenario = "thermal_drift"
    state.run = load_scenario(default_scenario)
    state.replay_idx = 0
    logger.info("Scenario loaded", extra={
        "scenario": default_scenario,
        "samples": len(state.run.process_data),
        "sessions": len(state.run.sessions),
        "recommendations": len(state.run.recommendations),
    })

    # Start the replay loop
    asyncio.create_task(replay_loop())

    # Optional: bridge a real OPC-UA server into the live tag stream.
    # Enabled when AXION_OPCUA_ENABLED is truthy. Coexists with the simulator
    # replay — the OPC-UA samples don't currently overwrite the active scenario,
    # they're observable via /api/integration/opcua/status. Wiring them into
    # state.run.process_data is a future block (T2).
    async def _on_opcua_sample(sample):
        state.opcua_buffer.append(sample)

    state.opcua = IntegrationService.from_env(on_sample=_on_opcua_sample)
    if state.opcua is not None:
        try:
            await state.opcua.start()
            logger.info("OPC-UA integration started", extra={
                "endpoint": state.opcua.tag_map.server.endpoint,
                "n_tags":   len(state.opcua.tag_map.tags),
            })
        except Exception as e:
            logger.error("OPC-UA integration start failed", extra={"error": str(e)})


# =============================================================================
# Replay loop — advances the clock and broadcasts events
# =============================================================================

async def replay_loop():
    """Background task that advances replay_idx and pushes WebSocket updates."""
    while True:
        await asyncio.sleep(REPLAY_TICK_SECONDS)

        # Live mode: rebuild process_data from the OPC-UA ring buffer and
        # park the cursor at the latest sample. Replay control still works
        # (pause/resume), but seek and progress no longer apply.
        if state.data_source == "opcua":
            await _refresh_live_run()
            if state.run is None or not state.replay_running:
                continue
        elif state.run is None or not state.replay_running:
            continue

        # How many simulation samples to advance per tick?
        # Process data is sampled at 60-second intervals. With speed=60×,
        # 1 wall-clock second should advance 60 simulation seconds = 1 sample.
        samples_per_tick = max(1, int(state.replay_speed / 60.0))
        new_idx = min(
            state.replay_idx + samples_per_tick,
            len(state.run.process_data) - 1
        )

        # Find new events that have crossed the clock since last tick
        old_ts = state.run.process_data["timestamp"].iloc[state.replay_idx]
        new_ts = state.run.process_data["timestamp"].iloc[new_idx]
        new_events = events_in_window(old_ts, new_ts)

        state.replay_idx = new_idx

        # Measure any operator outcomes whose evaluation time has now passed
        new_outcomes_msgs = _measure_due_operator_outcomes(new_ts)
        if new_outcomes_msgs:
            new_events = {**new_events, "operator_outcomes": new_outcomes_msgs}

        # Broadcast: current sample + any new events
        message = {
            "type":  "tick",
            "ts":    new_ts.isoformat(),
            "idx":   state.replay_idx,
            "total": len(state.run.process_data),
            "sample": current_sample_dict(),
            "new_events": new_events,
        }
        await broadcast(message)


def events_in_window(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Return any sessions/recommendations/decisions/outcomes inside (start, end]."""
    if state.run is None:
        return {}

    new_recs = []
    for rec in state.run.recommendations:
        if start < rec.timestamp <= end:
            summary = rec_to_summary(rec)
            new_recs.append(summary)
            recommendations_total.labels(
                rule=rec.rule_fired or "unknown",
                urgency=rec.urgency.value,
                scenario=state.run.scenario,
            ).inc()
            # Fire webhook for above-threshold recommendations (no-op if disabled)
            if state.webhook.enabled:
                payload = {
                    "id":         rec.id,
                    "timestamp":  rec.timestamp.isoformat(),
                    "urgency":    rec.urgency.value,
                    "rule_fired": rec.rule_fired,
                    "diagnosis":  rec.diagnosis,
                    "action":     summary.get("action", {}).get("description"),
                }
                state.webhook.notify(payload, scenario=state.run.scenario)

    new_decs = []
    for d in state.run.decisions:
        if start < d.timestamp <= end:
            new_decs.append({
                "id":             d.id,
                "rec_id":         d.recommendation_id,
                "ts":             d.timestamp.isoformat(),
                "status":         d.status.value,
                "operator_id":    d.operator_id,
                "justification":  d.justification,
            })

    new_outcomes = []
    for o in state.run.outcomes:
        if start < o.timestamp <= end:
            new_outcomes.append({
                "id":               o.id,
                "rec_id":           o.recommendation_id,
                "ts":               o.timestamp.isoformat(),
                "quality_score":    o.quality_score,
                "notes":            o.notes,
            })

    return {
        "recommendations": new_recs,
        "decisions":       new_decs,
        "outcomes":        new_outcomes,
    }


async def _refresh_live_run() -> None:
    """Materialize the OPC-UA ring buffer as state.run.process_data.

    Builds a minimal ScenarioRun that reuses the in-memory analytics + rules
    from startup. We re-run the analytical pipeline only when the buffer has
    grown enough to make it worthwhile (every 30 new samples).
    """
    df = state.opcua_buffer.to_dataframe()
    if df.empty:
        return

    needs_pipeline = (
        state.run is None
        or state.run.scenario != "live"
        or len(df) - len(state.run.process_data) >= 30
    )

    if not needs_pipeline:
        # Cheap path: just refresh the dataframe and advance the cursor
        state.run.process_data = df
        state.replay_idx = len(df) - 1
        return

    try:
        sessions = state.ae.run_sessions(df) if state.ae is not None else []
        recs     = state.re.generate(sessions, df) if state.re is not None else []
    except Exception as e:
        logger.error("Live pipeline failed", extra={"error": str(e)})
        sessions, recs = [], []

    state.run = ScenarioRun(
        scenario="live",
        process_data=df,
        sessions=sessions,
        recommendations=recs,
        decisions=[],
        executions=[],
        outcomes=[],
        performance=state.run.performance if state.run is not None else PerformanceTracker(),
        rec_df=recommendations_to_dataframe(recs),
        dec_df=pd.DataFrame(),
    )
    state.replay_idx = len(df) - 1


def _build_operator_overrides() -> Dict[str, OperatorOverride]:
    """Convert state.operator_overrides dicts into OperatorOverride objects."""
    out: Dict[str, OperatorOverride] = {}
    for rec_id, info in state.operator_overrides.items():
        ts_str = info.get("decision_ts")
        if not ts_str:
            continue
        out[rec_id] = OperatorOverride(
            rec_id=rec_id,
            status=info.get("status", "accepted"),
            justification=info.get("justification", ""),
            decision_ts=pd.Timestamp(ts_str),
        )
    return out


def _measure_due_operator_outcomes(now_ts: pd.Timestamp) -> List[dict]:
    """Run the operator-outcome tracker for every override whose measurement
    time has passed. Returns wire-format dicts for newly-measured outcomes."""
    if state.run is None or not state.operator_overrides:
        return []

    overrides = _build_operator_overrides()
    if not overrides:
        return []

    recs_by_id = {r.id: r for r in state.run.recommendations}
    process_data = state.run.process_data.iloc[: state.replay_idx + 1]

    new_outcomes = measure_pending(
        overrides=overrides,
        recs_by_id=recs_by_id,
        now_ts=now_ts,
        process_data=process_data,
        already_measured=set(state.operator_outcomes.keys()),
    )

    out: List[dict] = []
    for outcome in new_outcomes:
        rec = recs_by_id.get(outcome.recommendation_id)
        if rec is None:
            continue
        summary = outcome_summary_dict(outcome, rec)
        state.operator_outcomes[rec.id] = summary
        # Feed the performance tracker so per-rule confidence reflects UI decisions
        try:
            state.run.performance.record_outcome(rec, outcome)
        except Exception as e:
            logger.error("Performance tracker update failed",
                         extra={"rec_id": rec.id, "error": str(e)})
        out.append(summary)
    return out


# =============================================================================
# Conversion helpers
# =============================================================================

def current_sample_dict() -> dict:
    """Snapshot of the process state at the current replay index."""
    if state.run is None:
        return {}
    row = state.run.process_data.iloc[state.replay_idx]
    snapshot = {
        "timestamp":        row["timestamp"].isoformat(),
        "cstr": {
            "T_R_C":        float(row["cstr.T_R_C"]),
            "T_J_C":        float(row["cstr.T_J_C"]),
            "C_A":          float(row["cstr.C_A"]),
            "conversion":   float(row["cstr.conversion"]),
            "F_feed":       float(row["cstr.F_feed"]),
            "F_cool":       float(row["cstr.F_cool"]),
            "P_R":          float(row["cstr.P_R"]),
        },
        "column": {
            "purity_B":     float(row["column.purity_B"]),
            "x_D":          float(row["column.x_D"]),
            "T_top_C":      float(row["column.T_top_C"]),
            "T_bot_C":      float(row["column.T_bot_C"]),
            "RR":           float(row["column.RR"]),
            "Q_reb_kW":     float(row["column.Q_reb_kW"]),
            "F_vap_kgh":    float(row["column.F_vap_kgh"]),
        },
    }
    # Add soft sensor prediction for purity if available
    if state.soft_sensor is not None:
        try:
            X = pd.DataFrame([{f: row[f] for f in state.soft_sensor.feature_names}])
            pred, std = state.soft_sensor.predict_with_confidence(X)
            measured = float(row["column.purity_B"])
            predicted = float(pred[0])
            uncertainty = float(std[0])
            # Agreement: 1.0 = perfect, 0.0 = diverged by 5+ percentage points
            residual = abs(measured - predicted)
            agreement = max(0.0, 1.0 - residual / 5.0)
            snapshot["soft_sensor"] = {
                "purity_B_predicted":  predicted,
                "purity_B_std":        uncertainty,
                "purity_B_measured":   measured,
                "residual":            measured - predicted,
                "agreement":           agreement,
            }
        except Exception:
            pass
    return snapshot


def rec_to_summary(rec) -> dict:
    """Serialize a Recommendation for the wire format."""
    return {
        "id":              rec.id,
        "ts":              rec.timestamp.isoformat(),
        "urgency":         rec.urgency.value,
        "priority_score":  rec.priority_score,
        "confidence":      rec.confidence,
        "diagnosis":       rec.diagnosis,
        "probable_cause":  rec.probable_cause,
        "rule_fired":      rec.rule_fired,
        "affected_variables": rec.affected_variables,
        "action": {
            "type":            rec.action.type.value,
            "description":     rec.action.description,
            "target_variable": rec.action.target_variable,
            "current_value":   rec.action.current_value,
            "proposed_value":  rec.action.proposed_value,
            "adjustment":      rec.action.adjustment,
            "units":           rec.action.units,
        },
        "expected_impact": [
            {
                "variable":               imp.variable,
                "current_value":          imp.current_value,
                "predicted_value":        imp.predicted_value,
                "time_to_effect_minutes": imp.time_to_effect_minutes,
                "description":            imp.description,
            }
            for imp in rec.expected_impact
        ],
    }


# =============================================================================
# WebSocket
# =============================================================================

async def broadcast(message: dict):
    """Send a JSON message to all connected WebSocket clients."""
    if not state.clients:
        return
    payload = json.dumps(message)
    dead = set()
    for ws in list(state.clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        state.clients.discard(ws)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    state.clients.add(websocket)
    websocket_connections.set(len(state.clients))
    # Send an initial snapshot
    await websocket.send_text(json.dumps({
        "type":  "snapshot",
        "ts":    state.run.process_data["timestamp"].iloc[state.replay_idx].isoformat() if state.run else None,
        "idx":   state.replay_idx,
        "total": len(state.run.process_data) if state.run else 0,
        "sample": current_sample_dict(),
        "scenario": state.run.scenario if state.run else None,
    }))
    try:
        while True:
            # Keep the connection alive; ignore incoming messages for the MVP
            await websocket.receive_text()
    except WebSocketDisconnect:
        state.clients.discard(websocket)
        websocket_connections.set(len(state.clients))
    except Exception:
        state.clients.discard(websocket)
        websocket_connections.set(len(state.clients))


# =============================================================================
# REST endpoints
# =============================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "scenario": state.run.scenario if state.run else None,
        "samples_total": len(state.run.process_data) if state.run else 0,
        "replay_idx": state.replay_idx,
    }


@app.get("/api/scenarios")
async def get_scenarios():
    return {
        "available": list_scenarios(),
        "active":    state.run.scenario if state.run else None,
    }


@app.post("/api/scenarios/upload")
async def upload_scenario(
    file: UploadFile = File(...),
    name: str = Form("custom_run"),
    activate: bool = Form(True),
):
    """
    Upload a custom scenario CSV.

    The file must contain the standard process columns. On success, it is
    written to data/<name>.csv (overwriting any existing file with the same
    name) and optionally loaded as the active scenario.
    """
    from api.uploads import validate_scenario_name, validate_csv_bytes

    name_err = validate_scenario_name(name)
    if name_err:
        raise HTTPException(status_code=400, detail=name_err)

    raw = await file.read()
    result = validate_csv_bytes(raw)
    if not result.ok:
        raise HTTPException(
            status_code=400,
            detail={"errors": result.errors, "warnings": result.warnings},
        )

    target = DATA_DIR / f"{name}.csv"
    target.write_bytes(raw)
    logger.info("Custom scenario uploaded", extra={
        "scenario_name": name,
        "rows": result.n_rows,
        "cols": result.n_cols,
    })

    response: dict = {
        "ok":       True,
        "scenario": name,
        "n_rows":   result.n_rows,
        "n_cols":   result.n_cols,
        "warnings": result.warnings,
        "activated": False,
    }

    if activate:
        try:
            state.run = load_scenario(name)
            state.replay_idx = 0
            state.operator_overrides.clear()
            state.webhook.reset()
            state.pareto_cache = None
            await broadcast({"type": "scenario_changed", "scenario": name})
            response["activated"] = True
        except Exception as e:
            logger.error("Failed to activate uploaded scenario",
                         extra={"scenario_name": name, "error": str(e)})
            response["activation_error"] = str(e)

    return response


@app.post("/api/scenarios/select")
async def select_scenario(payload: dict):
    """Switch the active scenario. Body: {"scenario": "thermal_drift"}."""
    scn = payload.get("scenario")
    if scn not in list_scenarios():
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {scn}")
    state.run = load_scenario(scn)
    state.replay_idx = 0
    state.operator_overrides.clear()
    state.operator_outcomes.clear()
    state.webhook.reset()
    await broadcast({"type": "scenario_changed", "scenario": scn})
    return {"ok": True, "scenario": scn, "samples_total": len(state.run.process_data)}


@app.get("/api/state")
async def get_state():
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")
    return current_sample_dict()


@app.get("/api/process/recent")
async def get_recent(samples: int = 240):
    """Return the last N samples up to the current replay index."""
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")
    end = state.replay_idx + 1
    start = max(0, end - samples)
    sub = state.run.process_data.iloc[start:end]
    return {
        "tags": ["cstr.T_R_C", "cstr.T_J_C", "cstr.C_A", "cstr.F_cool",
                 "column.purity_B", "column.RR", "column.T_bot_C", "column.Q_reb_kW"],
        "data": [
            {
                "ts": row["timestamp"].isoformat(),
                "cstr.T_R_C":      float(row["cstr.T_R_C"]),
                "cstr.T_J_C":      float(row["cstr.T_J_C"]),
                "cstr.C_A":        float(row["cstr.C_A"]),
                "cstr.F_cool":     float(row["cstr.F_cool"]),
                "column.purity_B": float(row["column.purity_B"]),
                "column.RR":       float(row["column.RR"]),
                "column.T_bot_C":  float(row["column.T_bot_C"]),
                "column.Q_reb_kW": float(row["column.Q_reb_kW"]),
            }
            for _, row in sub.iterrows()
        ],
    }


@app.get("/api/recommendations")
async def get_recommendations(limit: int = 50, only_active: bool = True):
    """List recommendations issued up to current replay time."""
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")
    now_ts = state.run.process_data["timestamp"].iloc[state.replay_idx]
    visible = [r for r in state.run.recommendations if r.timestamp <= now_ts]
    visible.sort(key=lambda r: r.timestamp, reverse=True)
    out = []
    for rec in visible[:limit]:
        d = rec_to_summary(rec)
        # Attach decision status if known
        decision_for_rec = next(
            (x for x in state.run.decisions if x.recommendation_id == rec.id),
            None,
        )
        if rec.id in state.operator_overrides:
            d["status"] = state.operator_overrides[rec.id]["status"]
            d["operator_id"] = "operator_ui"
            d["justification"] = state.operator_overrides[rec.id].get("justification", "")
        elif decision_for_rec:
            d["status"] = decision_for_rec.status.value
            d["operator_id"] = decision_for_rec.operator_id
            d["justification"] = decision_for_rec.justification
        else:
            d["status"] = "pending"
            d["operator_id"] = None
            d["justification"] = ""
        out.append(d)
    return {"recommendations": out, "total": len(visible)}


@app.get("/api/recommendations/{rec_id}")
async def get_recommendation_detail(rec_id: str):
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")
    rec = next((r for r in state.run.recommendations if r.id == rec_id), None)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found")
    return rec_to_summary(rec)


@app.post("/api/recommendations/{rec_id}/decide")
async def decide_recommendation(rec_id: str, payload: dict):
    """Operator decision via UI. Body: {action, justification}."""
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")
    rec = next((r for r in state.run.recommendations if r.id == rec_id), None)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found")
    action = payload.get("action")  # 'accept' | 'reject' | 'modify'
    if action not in ("accept", "reject", "modify"):
        raise HTTPException(status_code=400, detail="action must be accept|reject|modify")
    status_map = {
        "accept": "accepted",
        "reject": "rejected",
        "modify": "modified",
    }
    justification = payload.get("justification", "Operator action via UI")
    new_status = status_map[action]
    decision_ts = state.run.process_data["timestamp"].iloc[state.replay_idx]
    state.operator_overrides[rec.id] = {
        "status":        new_status,
        "justification": justification,
        "decision_ts":   decision_ts.isoformat(),
    }

    if state.db:
        try:
            state.db.insert_decision(rec.id, new_status, justification)
            state.db.update_recommendation_status(rec.id, new_status)
        except Exception as e:
            logger.error("DB decision persist failed", extra={"error": str(e)})

    operator_decisions_total.labels(status=new_status).inc()

    await broadcast({
        "type":   "decision_recorded",
        "rec_id": rec.id,
        "status": new_status,
    })
    return {"ok": True, "rec_id": rec.id, "status": new_status}


@app.get("/api/decisions")
async def get_decisions(limit: int = 100):
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")
    now_ts = state.run.process_data["timestamp"].iloc[state.replay_idx]
    visible = [d for d in state.run.decisions if d.timestamp <= now_ts]
    visible.sort(key=lambda d: d.timestamp, reverse=True)
    out = []
    for d in visible[:limit]:
        out.append({
            "id":               d.id,
            "rec_id":           d.recommendation_id,
            "ts":               d.timestamp.isoformat(),
            "status":           d.status.value,
            "operator_id":      d.operator_id,
            "justification":    d.justification,
        })
    return {"decisions": out, "total": len(visible)}


@app.get("/api/performance")
async def get_performance():
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")
    df = state.run.performance.summary_dataframe()
    if df.empty:
        return {"by_rule": []}
    return {"by_rule": df.to_dict(orient="records")}


@app.get("/api/replay/status")
async def replay_status():
    if state.run is None:
        return {
            "running": False, "idx": 0, "total": 0,
            "speed": state.replay_speed, "data_source": state.data_source,
        }
    return {
        "running":    state.replay_running,
        "idx":        state.replay_idx,
        "total":      len(state.run.process_data),
        "progress":   state.replay_idx / max(1, len(state.run.process_data) - 1),
        "speed":      state.replay_speed,
        "ts":         state.run.process_data["timestamp"].iloc[state.replay_idx].isoformat(),
        "scenario":   state.run.scenario,
        "data_source": state.data_source,
    }


# =============================================================================
# Data versioning endpoints (Bloque Z)
# =============================================================================

SNAPSHOTS_DIR = DATA_DIR / ".versions"


@app.get("/api/data/snapshots")
async def list_data_snapshots():
    """List every data snapshot manifest under data/.versions/."""
    from data_versioning import list_snapshots
    snaps = list_snapshots(SNAPSHOTS_DIR)
    return {
        "count": len(snaps),
        "snapshots": [
            {
                "snapshot_id": s.snapshot_id,
                "created_at":  s.created_at,
                "message":     s.message,
                "n_files":     len(s.files),
            }
            for s in snaps
        ],
    }


@app.get("/api/data/snapshots/{snapshot_id}")
async def get_data_snapshot(snapshot_id: str):
    """Return a snapshot's full manifest (all file hashes + metadata)."""
    from data_versioning import load_snapshot
    try:
        snap = load_snapshot(SNAPSHOTS_DIR, snapshot_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404,
                              detail=f"Snapshot {snapshot_id!r} not found")
    return snap.to_dict()


@app.get("/api/data/snapshots/{snapshot_id}/verify")
async def verify_data_snapshot(snapshot_id: str):
    """Re-hash data/ and compare against the snapshot. Returns ok=True when
    every file still matches; otherwise lists missing/extra/mismatched."""
    from data_versioning import load_snapshot, verify_snapshot
    try:
        snap = load_snapshot(SNAPSHOTS_DIR, snapshot_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404,
                              detail=f"Snapshot {snapshot_id!r} not found")
    return verify_snapshot(snap, DATA_DIR)


# =============================================================================
# Metrics endpoint (Bloque W)
# =============================================================================

from fastapi import Response


@app.get("/api/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus text-format exposition. Always public so a scraper
    without an API key can poll it."""
    body, content_type = render_prometheus()
    return Response(content=body, media_type=content_type)


# =============================================================================
# Auth endpoints (JWT — Bloque V)
# =============================================================================

def _require_jwt_secret() -> str:
    secret = jwt_secret()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="JWT auth not configured. Set AXION_JWT_SECRET.",
        )
    return secret


def _require_users_repo() -> UserRepository:
    if state.db is None:
        raise HTTPException(
            status_code=503,
            detail="Users DB not available. Configure AXION_DB_URL.",
        )
    return UserRepository(state.db)


@app.post("/api/auth/login")
async def auth_login(payload: dict):
    """Exchange email + password for an access + refresh JWT pair."""
    secret = _require_jwt_secret()
    repo   = _require_users_repo()

    email    = (payload or {}).get("email", "").strip().lower()
    password = (payload or {}).get("password", "")
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password required")

    try:
        user = repo.get_by_email(email)
    except Exception as e:
        logger.error("User lookup failed", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail="User store unavailable")

    if user is None or not user.active:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access, refresh, exp_in = issue_token_pair(
        secret=secret, sub=user.email, uid=user.id, role=user.role,
    )
    logger.info("User logged in", extra={"uid": user.id, "role": user.role})
    return {
        "access_token":  access,
        "refresh_token": refresh,
        "token_type":    "bearer",
        "expires_in":    exp_in,
        "user":          user.to_public_dict(),
    }


@app.post("/api/auth/refresh")
async def auth_refresh(payload: dict):
    """Trade a refresh token for a fresh access + refresh pair."""
    secret = _require_jwt_secret()
    repo   = _require_users_repo()

    rt = (payload or {}).get("refresh_token", "")
    if not rt:
        raise HTTPException(status_code=400, detail="refresh_token required")

    try:
        claims = decode_token(rt, secret, expected_type="refresh")
    except TokenExpiredError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except (InvalidTokenError, TokenTypeMismatchError, AuthError) as e:
        raise HTTPException(status_code=401, detail=f"Invalid refresh token: {e}")

    # Re-check the user — disabled users should not be refreshable
    try:
        user = repo.get_by_id(claims.uid)
    except Exception as e:
        logger.error("User lookup failed in refresh", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail="User store unavailable")
    if user is None or not user.active:
        raise HTTPException(status_code=401, detail="User no longer active")

    access, refresh, exp_in = issue_token_pair(
        secret=secret, sub=user.email, uid=user.id, role=user.role,
    )
    return {
        "access_token":  access,
        "refresh_token": refresh,
        "token_type":    "bearer",
        "expires_in":    exp_in,
    }


@app.get("/api/auth/me")
async def auth_me(request: Request):
    """Return the currently authenticated user (read from middleware state)."""
    role  = getattr(request.state, "role", None)
    email = getattr(request.state, "user_email", None)
    uid   = getattr(request.state, "user_id", None)
    if role is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"email": email, "uid": uid, "role": role}


# =============================================================================
# Process profile endpoints (multi-process)
# =============================================================================

@app.get("/api/profile")
async def get_active_profile():
    """Return metadata for the currently active process profile."""
    return {
        "active":     active_profile().name,
        "available":  list_profiles(),
        "profile":    active_profile().to_dict(),
    }


@app.post("/api/profile/select")
async def select_profile(payload: dict):
    """Switch the active process profile.

    Body: {"profile": "pilot" | "batch_reactor"}.

    Sets the AXION_PROCESS_PROFILE env var so subsequent calls resolve to
    the new profile. Loaded scenarios are NOT reloaded automatically — the
    caller is expected to choose a scenario from the new profile via
    /api/scenarios/select. Returns 400 for unknown profile names.
    """
    name = (payload or {}).get("profile", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="profile required")
    try:
        new_profile = get_profile(name)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile {name!r}. Available: {list_profiles()}",
        )
    os.environ["AXION_PROCESS_PROFILE"] = name
    logger.info("Process profile switched", extra={"profile": name})

    # Rebuild engines + drift detector against the new profile's baseline.
    # Wrapped in try/except so a misconfigured profile doesn't kill the
    # server — the old engines remain in place.
    try:
        baseline_csv = _baseline_csv_for(new_profile)
        df_train = pd.read_csv(baseline_csv)
        state.ae = AnalyticalEngine(
            tags=new_profile.tag_names,
            operational_limits=new_profile.operational_limits or None,
            training_fraction=1.0,
            warmup_minutes=15.0,
        )
        state.ae.fit(df_train)
        state.re = RecommendationEngine(
            rules=new_profile.load_rules() or None,
            operational_limits=new_profile.operational_limits or None,
        )
        drift_features = new_profile.feature_cols or PILOT_PURITY_FEATURES
        state.drift_detector = DriftDetector(features=drift_features).fit(df_train)
        # Reload first available scenario for this profile so the dashboard
        # has data to render
        scenarios = [s for s in new_profile.scenarios
                     if (DATA_DIR / f"{s}.csv").exists()]
        if scenarios:
            state.run = load_scenario(scenarios[0])
            state.replay_idx = 0
            state.operator_overrides.clear()
            state.operator_outcomes.clear()
            state.webhook.reset()
    except Exception as e:
        logger.error("Profile rebuild failed — keeping previous engines",
                     extra={"profile": name, "error": str(e)})
        return {"ok": True, "profile": new_profile.to_dict(),
                "warning": f"Engines kept on previous profile: {e}"}

    await broadcast({"type": "profile_changed", "profile": name})
    return {"ok": True, "profile": new_profile.to_dict()}


# =============================================================================
# Data source switch (replay ↔ opcua)
# =============================================================================

@app.post("/api/data-source/select")
async def select_data_source(payload: dict):
    """Switch between scenario replay and the live OPC-UA bridge.

    Body: {"source": "replay" | "opcua"}.

    Switching to "opcua" requires AXION_OPCUA_ENABLED to be set so the
    `IntegrationService` is running. Switching to "replay" reloads the last
    active scenario CSV and resets the replay clock.
    """
    src = (payload or {}).get("source", "").strip().lower()
    if src not in ("replay", "opcua"):
        raise HTTPException(status_code=400, detail="source must be 'replay' or 'opcua'")

    if src == "opcua":
        if state.opcua is None:
            raise HTTPException(
                status_code=503,
                detail="OPC-UA not configured. Set AXION_OPCUA_ENABLED=true and restart.",
            )
        state.data_source = "opcua"
        state.operator_overrides.clear()
        state.operator_outcomes.clear()
        state.webhook.reset()
        await _refresh_live_run()
        await broadcast({"type": "data_source_changed", "source": "opcua"})
        return {"ok": True, "source": "opcua",
                "buffer_samples": len(state.opcua_buffer)}

    # src == "replay"
    state.data_source = "replay"
    state.operator_overrides.clear()
    state.operator_outcomes.clear()
    state.webhook.reset()
    # Restore the last scenario CSV (default to thermal_drift if no run)
    scenario = state.run.scenario if state.run and state.run.scenario != "live" else "thermal_drift"
    state.run = load_scenario(scenario)
    state.replay_idx = 0
    await broadcast({"type": "data_source_changed", "source": "replay",
                     "scenario": scenario})
    return {"ok": True, "source": "replay", "scenario": scenario}


@app.get("/api/data-source/status")
async def data_source_status():
    return {
        "source":         state.data_source,
        "opcua_enabled":  state.opcua is not None,
        "opcua_connected": state.opcua.status.connected if state.opcua else False,
        "buffer_samples": len(state.opcua_buffer),
    }


@app.get("/api/soft_sensor/purity")
async def soft_sensor_purity(samples: int = 240):
    """
    Return soft sensor prediction + uncertainty band for the last N samples
    of the active scenario, up to the current replay index. For overlay on
    the purity chart.
    """
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")
    if state.soft_sensor is None:
        return {"available": False, "data": []}
    end = state.replay_idx + 1
    start = max(0, end - samples)
    sub = state.run.process_data.iloc[start:end]
    try:
        X = sub[state.soft_sensor.feature_names]
        preds, stds = state.soft_sensor.predict_with_confidence(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Soft sensor inference failed: {e}")

    data = [
        {
            "ts":        row["timestamp"].isoformat(),
            "measured":  float(row["column.purity_B"]),
            "predicted": float(preds[i]),
            "std":       float(stds[i]),
        }
        for i, (_, row) in enumerate(sub.iterrows())
    ]
    meta = {
        "features":  state.soft_sensor.feature_names,
        "target":    state.soft_sensor.target_name,
    }
    if state.soft_sensor.metrics is not None:
        meta.update({
            "mae":  state.soft_sensor.metrics.mae,
            "r2":   state.soft_sensor.metrics.r2,
        })
    return {"available": True, "meta": meta, "data": data}


# =============================================================================
# Optimization endpoints (Pareto front of operating points)
# =============================================================================

def _compute_pareto(disturbances: dict, n_generations: int = 30, population: int = 60) -> dict:
    """Run NSGA-II for the current operating context and return a serializable
    Pareto front."""
    if state.surrogate is None:
        raise RuntimeError("Surrogate not loaded")

    bounds = {
        "column.RR":   (3.0, 7.5),
        "cstr.F_cool": (0.10, 0.55),
        "cstr.F_feed": (1.7, 2.3),
    }
    objectives = [
        PurityObjective(weight=1.0, spec=98.5),
        EnergyObjective(weight=1.0),
        ProductionObjective(weight=0.7),
        StabilityObjective(weight=0.3),
    ]
    optimizer = NSGA2Optimizer(
        surrogate=state.surrogate,
        objectives=objectives,
        bounds=bounds,
        fixed_inputs=disturbances,
        seed=42,
    )
    front = optimizer.run(n_generations=n_generations, population_size=population)
    # Sort by purity for stable display
    front_sorted = sorted(front, key=lambda p: p.objectives.get("purity", 0))
    return {
        "available": True,
        "objectives": [
            {"name": o.name, "direction": o.direction.value, "units": o.units}
            for o in objectives
        ],
        "decision_variables": list(bounds.keys()),
        "bounds": {k: list(v) for k, v in bounds.items()},
        "fixed":  disturbances,
        "front": [
            {
                "inputs":     p.inputs,
                "kpis":       p.kpis,
                "objectives": p.objectives,
            }
            for p in front_sorted
        ],
    }


@app.get("/api/optimization/pareto")
async def optimization_pareto(refresh: bool = False):
    """
    Return the Pareto front of operating points balancing purity, energy,
    production and stability. Computed for the current process disturbance
    context (C_A, T_feed). Cached per session — pass refresh=true to recompute.
    """
    if state.surrogate is None:
        raise HTTPException(status_code=503, detail="Surrogate not available")
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")

    if state.pareto_cache is None or refresh:
        # Take disturbance context from the current process snapshot
        row = state.run.process_data.iloc[state.replay_idx]
        disturbances = {
            "cstr.C_A":      float(row["cstr.C_A"]),
            "cstr.T_feed_C": float(row["cstr.T_feed_C"]),
        }
        try:
            state.pareto_cache = _compute_pareto(disturbances)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    # Add the nominal operating point for visual reference
    nominal_inputs = {
        "column.RR":     5.5,
        "cstr.F_cool":   0.30,
        "cstr.F_feed":   2.0,
        "cstr.C_A":      state.pareto_cache["fixed"]["cstr.C_A"],
        "cstr.T_feed_C": state.pareto_cache["fixed"]["cstr.T_feed_C"],
    }
    nominal_kpis = state.surrogate.predict_one(**nominal_inputs)
    state.pareto_cache["nominal"] = {"inputs": nominal_inputs, "kpis": nominal_kpis}

    # Add the current operating point (real values from process data)
    row = state.run.process_data.iloc[state.replay_idx]
    current_inputs = {
        "column.RR":     float(row["column.RR"]),
        "cstr.F_cool":   float(row["cstr.F_cool"]),
        "cstr.F_feed":   float(row["cstr.F_feed"]),
        "cstr.C_A":      float(row["cstr.C_A"]),
        "cstr.T_feed_C": float(row["cstr.T_feed_C"]),
    }
    state.pareto_cache["current"] = {
        "inputs": current_inputs,
        "kpis": {
            "column.purity_B": float(row["column.purity_B"]),
            "column.Q_reb_kW": float(row["column.Q_reb_kW"]),
            "cstr.conversion": float(row["cstr.conversion"]),
            "cstr.T_R_C":      float(row["cstr.T_R_C"]),
        },
    }

    return state.pareto_cache


@app.post("/api/optimization/predict")
async def optimization_predict(payload: dict):
    """
    Predict KPIs for a manually-specified operating point. Used by the UI
    when the operator drags the marker on the Pareto chart and wants to see
    the predicted KPIs at that point.
    Body: {"column.RR": 5.5, "cstr.F_cool": 0.30, "cstr.F_feed": 2.0}
    """
    if state.surrogate is None:
        raise HTTPException(status_code=503, detail="Surrogate not available")
    inputs = {k: float(v) for k, v in payload.items()
              if k in state.surrogate.inputs}
    # Fill missing from current state
    if state.run is not None:
        row = state.run.process_data.iloc[state.replay_idx]
        for k in state.surrogate.inputs:
            if k not in inputs:
                inputs[k] = float(row[k])
    return {"inputs": inputs, "kpis": state.surrogate.predict_one(**inputs)}


# =============================================================================
# Predictive endpoints (LSTM forecaster)
# =============================================================================

@app.get("/api/predictive/forecast")
async def predictive_forecast():
    """
    Return the LSTM-predicted future trajectory for all target variables
    starting from the current replay position. Includes per-horizon point
    predictions for the standard horizons [5, 15, 30, 60] minutes.
    """
    if state.forecaster is None:
        return {"available": False, "reason": "LSTM forecaster not loaded"}
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")

    end = state.replay_idx + 1
    sub = state.run.process_data.iloc[: end].copy()
    if len(sub) < state.forecaster.config.lookback_steps:
        return {"available": True, "warming_up": True,
                "needs_samples": state.forecaster.config.lookback_steps,
                "have_samples":  len(sub)}

    full = state.forecaster.predict_from_df(sub)
    if full is None:
        return {"available": True, "warming_up": True}

    sample_period = state.forecaster.config.sample_period_minutes
    now_ts = sub["timestamp"].iloc[-1]
    horizons_min = state.forecaster.config.horizons_minutes
    targets = state.forecaster.target_cols

    points = state.forecaster.predict_at_horizons(sub) or {}

    trajectories = {}
    for tgt, traj in full.items():
        trajectories[tgt] = [
            {
                "horizon_min": float((s + 1) * sample_period),
                "ts":          (now_ts + pd.Timedelta(minutes=(s + 1) * sample_period)).isoformat(),
                "predicted":   float(traj[s]),
            }
            for s in range(len(traj))
        ]

    current = {tgt: float(sub[tgt].iloc[-1]) for tgt in targets}

    meta = {
        "horizons_minutes":  horizons_min,
        "lookback_minutes":  state.forecaster.config.lookback_minutes,
        "targets":           targets,
    }

    return {
        "available":     True,
        "warming_up":    False,
        "now":           now_ts.isoformat(),
        "current":       current,
        "points":        points,
        "trajectories":  trajectories,
        "meta":          meta,
    }


@app.post("/api/replay/control")
async def replay_control(payload: dict):
    """Pause / resume / set speed / seek. Body: {action, speed?, idx?}"""
    action = payload.get("action")
    if action == "pause":
        state.replay_running = False
    elif action == "resume":
        state.replay_running = True
    elif action == "speed":
        state.replay_speed = float(payload.get("speed", REPLAY_SPEED_DEFAULT))
    elif action == "seek":
        if state.run is None:
            raise HTTPException(status_code=404, detail="No scenario loaded")
        idx = int(payload.get("idx", 0))
        state.replay_idx = max(0, min(idx, len(state.run.process_data) - 1))
    elif action == "restart":
        state.replay_idx = 0
        state.operator_overrides.clear()
    else:
        raise HTTPException(status_code=400, detail="unknown action")
    return {"ok": True, "running": state.replay_running, "idx": state.replay_idx,
            "speed": state.replay_speed}


# =============================================================================
# History endpoints (query TimescaleDB)
# =============================================================================

def _require_db():
    if state.db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Start TimescaleDB and set AXION_DB_URL.",
        )


@app.get("/api/history/scenarios")
async def history_scenarios():
    """List all scenarios that have been persisted to the DB."""
    _require_db()
    try:
        return {"scenarios": state.db.list_ingested_scenarios()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/samples")
async def history_samples(
    scenario: Optional[str] = None,
    from_ts:  Optional[str] = None,
    to_ts:    Optional[str] = None,
    tags:     Optional[str] = None,   # comma-separated dotted names
    limit:    int = 1000,
):
    """
    Query historical process samples from TimescaleDB.

    - scenario: filter by scenario name (default: all)
    - from_ts / to_ts: ISO-8601 timestamps (e.g. 2026-01-01T00:00:00)
    - tags: comma-separated sensor tags (e.g. cstr.T_R_C,column.purity_B)
    - limit: max rows returned (default 1000, max 10000)
    """
    _require_db()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    try:
        rows = state.db.query_samples(
            scenario=scenario or (state.run.scenario if state.run else None),
            from_ts=from_ts,
            to_ts=to_ts,
            tags=tag_list,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"count": len(rows), "data": rows}


@app.get("/api/history/recommendations")
async def history_recommendations(
    scenario: Optional[str] = None,
    from_ts:  Optional[str] = None,
    to_ts:    Optional[str] = None,
    status:   Optional[str] = None,   # comma-separated: pending,accepted,rejected
    urgency:  Optional[str] = None,   # comma-separated: low,medium,high,critical
    limit:    int = 200,
):
    """Query historical recommendations from TimescaleDB."""
    _require_db()
    status_list  = [s.strip() for s in status.split(",")]  if status  else None
    urgency_list = [u.strip() for u in urgency.split(",")] if urgency else None
    try:
        rows = state.db.query_recommendations(
            scenario=scenario or (state.run.scenario if state.run else None),
            from_ts=from_ts,
            to_ts=to_ts,
            status=status_list,
            urgency=urgency_list,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"count": len(rows), "data": rows}


@app.get("/api/history/decisions")
async def history_decisions(
    scenario: Optional[str] = None,
    from_ts:  Optional[str] = None,
    to_ts:    Optional[str] = None,
    limit:    int = 200,
):
    """Query historical operator decisions from TimescaleDB."""
    _require_db()
    try:
        rows = state.db.query_decisions(
            scenario=scenario or (state.run.scenario if state.run else None),
            from_ts=from_ts,
            to_ts=to_ts,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"count": len(rows), "data": rows}


# =============================================================================
# Model status endpoint
# =============================================================================

@app.get("/api/models/status")
async def models_status():
    """Return the training status and evaluation metrics of deployed models.

    Reads the JSON metrics files written by the retraining pipeline. Safe to
    call at any time — returns {"status": "not_trained"} when files are absent.
    """
    result: Dict = {}

    # ── Soft sensor ──────────────────────────────────────────────────────────
    ss_metrics_path = MODELS_DIR / "purity_soft_sensor.metrics.json"
    ss_model_path   = MODELS_DIR / "purity_soft_sensor.joblib"

    if ss_metrics_path.exists():
        try:
            with open(ss_metrics_path) as fh:
                metrics = json.load(fh)
        except Exception:
            metrics = None
        result["soft_sensor"] = {
            "status":     "trained",
            "trained_at": datetime.fromtimestamp(
                ss_metrics_path.stat().st_mtime
            ).isoformat(),
            "metrics": metrics,
        }
    elif ss_model_path.exists():
        result["soft_sensor"] = {
            "status":     "trained",
            "trained_at": datetime.fromtimestamp(
                ss_model_path.stat().st_mtime
            ).isoformat(),
            "metrics": None,
        }
    else:
        result["soft_sensor"] = {"status": "not_trained", "metrics": None}

    # ── LSTM forecaster ───────────────────────────────────────────────────────
    lstm_dir = MODELS_DIR / "lstm_forecaster"
    if lstm_dir.exists():
        result["lstm_forecaster"] = {
            "status":     "trained",
            "trained_at": datetime.fromtimestamp(
                lstm_dir.stat().st_mtime
            ).isoformat(),
        }
    else:
        result["lstm_forecaster"] = {"status": "not_trained"}

    return result


# =============================================================================
# OPC-UA integration status
# =============================================================================

@app.get("/api/integration/opcua/status")
async def opcua_status():
    """Health snapshot of the live OPC-UA bridge. Always returns 200 — the
    `enabled` flag tells the UI whether the bridge was even configured."""
    if state.opcua is None:
        return {"enabled": False, "connected": False}
    return state.opcua.status.to_dict()


# =============================================================================
# Operator outcomes endpoint
# =============================================================================

@app.get("/api/outcomes/operator")
async def operator_outcomes(limit: int = 100):
    """Return outcomes measured for UI-recorded operator decisions.

    Each entry compares the recommendation's predicted impact to the actual
    process value observed `measurement_delay_min` minutes after the decision.
    """
    items = list(state.operator_outcomes.values())
    items.sort(key=lambda o: o.get("measured_at", ""), reverse=True)
    return {
        "count":    len(items),
        "outcomes": items[:limit],
    }


# =============================================================================
# Drift detection endpoint
# =============================================================================

@app.get("/api/drift/status")
async def drift_status(window: int = 240):
    """Compute distributional drift between the training reference and the
    last `window` samples up to the current replay index.

    Returns a per-feature PSI report plus an overall status:
        none / moderate / significant
    """
    if state.drift_detector is None or not state.drift_detector.fitted:
        return {"available": False, "reason": "Drift detector not fitted"}
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")

    end   = state.replay_idx + 1
    start = max(0, end - max(1, window))
    sub   = state.run.process_data.iloc[start:end]

    if len(sub) < 30:
        return {
            "available":   True,
            "warming_up":  True,
            "have_samples": len(sub),
            "needs_samples": 30,
        }

    try:
        report = state.drift_detector.score(sub)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift scoring failed: {e}")

    return {
        "available":  True,
        "warming_up": False,
        "scenario":   state.run.scenario,
        **report.to_dict(),
    }


# =============================================================================
# Webhook status endpoint
# =============================================================================

@app.get("/api/webhook/status")
async def webhook_status():
    """Return whether the webhook is enabled and its (sanitized) config."""
    wh = state.webhook
    return {
        "enabled":   wh.enabled,
        "threshold": wh.threshold,
        "format":    wh.fmt,
        "n_fired":   len(wh._fired),
    }


# =============================================================================
# Report endpoint
# =============================================================================

@app.get("/api/report/current", response_class=HTMLResponse)
async def report_current():
    """Generate and return a self-contained HTML process report for the current scenario."""
    if state.run is None:
        raise HTTPException(status_code=404, detail="No scenario loaded")

    from api.report import (
        kpi_summary, recommendations_summary, decisions_summary,
        sessions_summary, render_html,
    )

    # Map rec.id → {urgency, rule_id} for enriching decision dicts
    rec_lookup = {
        rec.id: {"urgency": rec.urgency.value, "rule_id": rec.rule_fired}
        for rec in state.run.recommendations
    }

    # Resolve each recommendation's current status
    rec_dicts = []
    for rec in state.run.recommendations:
        status = "pending"
        if rec.id in state.operator_overrides:
            status = state.operator_overrides[rec.id]["status"]
        else:
            dec = next(
                (d for d in state.run.decisions if d.recommendation_id == rec.id),
                None,
            )
            if dec:
                status = dec.status.value
        rec_dicts.append({
            "urgency":    rec.urgency.value,
            "rule_fired": rec.rule_fired,
            "status":     status,
            "timestamp":  rec.timestamp.isoformat(),
            "diagnosis":  rec.diagnosis,
        })

    dec_dicts = []
    for d in state.run.decisions:
        rec_info = rec_lookup.get(d.recommendation_id, {})
        dec_dicts.append({
            "status":        d.status.value,
            "justification": d.justification,
            "timestamp":     d.timestamp.isoformat(),
            "urgency":       rec_info.get("urgency", ""),
            "rule_id":       rec_info.get("rule_id", ""),
        })

    sess_dicts = [
        {
            "detector":      s.detector,
            "tag":           s.tag,
            "duration_min":  s.duration_minutes,
            "peak_severity": s.peak_severity.value,
            "start_time":    s.start_time.isoformat(),
        }
        for s in state.run.sessions
    ]

    perf_df = state.run.performance.summary_dataframe()
    perf_rows = perf_df.to_dict("records") if not perf_df.empty else []

    html = render_html(
        scenario=state.run.scenario,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        kpi=kpi_summary(state.run.process_data),
        rec_summary=recommendations_summary(rec_dicts),
        dec_summary=decisions_summary(dec_dicts),
        sess_summary=sessions_summary(sess_dicts),
        perf_rows=perf_rows,
        rec_log=rec_dicts,
    )
    return HTMLResponse(content=html)


# =============================================================================
# Serve the UI
# =============================================================================

if UI_DIR.exists():
    @app.get("/")
    async def serve_index():
        index_path = UI_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse({"detail": "UI not built; place index.html under /ui"})

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
