# Axion AI

Sistema autónomo de ingeniería de procesos para monitoreo, detección
de anomalías y forecasting predictivo en plantas industriales.

## Pilot

- **Proceso**: CSTR exotérmico + columna de destilación binaria
- **ICP**: Ingenieros de proceso y plant managers en química fina,
  petroquímica, farma, alimentos, papel, cemento, refinación.
  Empresas de 50–500 empleados.

## Stack técnico

- Python (simulator + ML)
- FastAPI (REST API + WebSocket)
- TimescaleDB (docker-compose + Alembic migrations + ingest scripts)
- LSTM (Keras/TensorFlow) con multi-target multi-horizon
- pytest para tests automatizados

## Estructura del repo

```
axion_simulator/
├── simulator/        # CSTR + columna + scenarios + sensores
├── analytics/        # SPC, PCA, Trend, Regime, FrozenSensor + sessions
├── recommendations/  # 10 reglas R1-R10 + engine
├── consensus/        # Modos ADVISOR/SEMI/AUTONOMOUS + safety gate
├── soft_sensor/      # GBR ensemble para purity_B
├── predictive/       # LSTM forecasting (5/15/30/60 min) + windowing
├── optimizer/        # NSGA-II + ProcessSurrogate + 4 objectives
├── integration/      # OPC-UA mock server + ingestion
├── api/              # FastAPI server (server.py)
├── ui/               # Dashboard HTML + JS
├── db/               # Alembic migrations (alembic.ini + env.py + versions/)
├── scripts/          # smoke_test.sh, ingest_csvs.py
├── tests/            # 269 tests unitarios (Bloque A completado)
└── data/             # 9 scenarios CSV (normal, thermal_drift, etc.)
```

## Estado de desarrollo

### v0.9 — DEMO-READY (no production-ready)

Listo para demos. Conocido pendiente:
- Validación contra ground truth del simulador (no plant data real)
- Sin auth/RBAC, sin CI/CD
- Sin MLflow para versioning de modelos

### Bloque A — COMPLETADO ✅

Tests automatizados con pytest:
- 269 tests unitarios (256 fast + 13 slow)
- Coverage ~81% global, ~94% en simulator/analytics/consensus
- `pytest.ini` + `conftest.py` con fixtures sintéticas y CSVs reales
- `pytest --run-slow` para incluir LSTM training y NSGA-II runs

### Bloque B — COMPLETADO ✅

Hardening de demo:
- [x] `axion_logging.py` con logger estructurado (JSON + pretty)
- [x] Reemplazar `print()` de producción en `api/server.py`,
      `integration/ingestion.py`, `integration/opcua_mock_server.py`
- [x] `Makefile` con targets: `test`, `test-slow`, `test-all`,
      `coverage`, `smoke`, `serve`, `clean`
- [x] `scripts/smoke_test.sh` con preflight, health poll, validación JSON

### Bloque C — COMPLETADO ✅

Persistencia real:
- [x] `docker-compose.yml` con TimescaleDB (pg16)
- [x] Migraciones Alembic (`db/versions/001_initial.py`):
      hypertable `process_samples`, tablas `scenarios`, `recommendations`, `decisions`
- [x] `scripts/ingest_csvs.py`: bulk insert idempotente, batch 1000 rows

### Bloque D — COMPLETADO ✅

Integración DB en el server:
- [x] `db/client.py`: `DbClient` con `insert_samples`, `upsert_recommendations`,
      `insert_decision`, `update_recommendation_status`
- [x] `api/server.py`: conecta a `AXION_DB_URL` en startup (graceful — sin DB
      el server arranca en modo memoria), persiste samples + recs al cargar
      escenario, persiste decisiones del operador vía API
- [x] Import condicional de TensorFlow en server y en `predictive/__init__.py`

### Bloque E — COMPLETADO ✅

Demo hardening:
- [x] `make demo`: levanta TimescaleDB, espera health, aplica migraciones, inicia server
- [x] `make demo-reset`: baja container + volumen, estado cero para repetir demo
- [x] `DEMO.md`: guía de una página para evaluadores externos (prerrequisitos,
      comandos, endpoints, escenarios, arquitectura)

### Bloque F — COMPLETADO ✅

CI/CD con GitHub Actions:
- [x] `.github/workflows/ci.yml`: job `fast` corre en todo push/PR sobre
      Python 3.11 y 3.12 (`make test`, ~60s con cache); job `slow` solo
      en push a main (`make test-slow`, LSTM + NSGA-II); coverage XML
      subido como artefacto en push a main

### Bloque G — COMPLETADO ✅

Auth básica con API key:
- [x] Middleware `api_key_middleware` en `api/server.py`: protege `/api/*`
      (excepto `/api/health`) y `/ws/*` con header `X-API-Key`; acepta
      `?api_key=` en query param para WebSocket; desactivado si
      `AXION_API_KEY` no está seteada
- [x] `tests/unit/test_api_auth.py`: 9 tests (auth desactivada, clave
      correcta/incorrecta/ausente, health siempre público, query param)
- [x] `.env.example`, `DEMO.md` y `smoke_test.sh` actualizados

### Bloque H — COMPLETADO ✅

Endpoints históricos (lectura desde TimescaleDB):
- [x] `db/client.py`: métodos `list_ingested_scenarios`, `query_samples`
      (filtros: scenario, from_ts, to_ts, tags, limit), `query_recommendations`
      (+ filtros status y urgency), `query_decisions` (JOIN con recommendations)
- [x] `api/server.py`: 4 endpoints `GET /api/history/{scenarios|samples|
      recommendations|decisions}` — devuelven 503 si DB no disponible
- [x] `tests/unit/test_api_history.py`: 15 tests con DbClient mockeado
      (503 sin DB, filtros parseados correctamente, shape de respuesta)

### Bloque I — COMPLETADO ✅

MLflow model versioning:
- [x] `axion_mlflow.py`: wrapper `Run` context manager que degrada a no-ops
      si MLflow no está instalado; emite un solo warning al iniciar
- [x] `examples/train_soft_sensor.py`: integrado con `Run("axion-soft-sensor")`;
      loguea params (n_ensemble, n_estimators, n_features), métricas overall
      (mae/rmse/r2), MAE por escenario, MAE holdout, y artifacts (joblib, CSV, PNG)
- [x] `examples/train_lstm_forecaster.py`: integrado con `Run("axion-lstm-forecaster")`;
      loguea params (lookback, horizons, units, dropout, epochs, batch_size),
      epoch loss curves vía `log_epoch_metrics`, MAE/R² por horizonte por target,
      y artifacts (model dir, figuras)
- [x] `tests/unit/test_axion_mlflow.py`: 30 tests (MLflow disponible vs no disponible,
      delegación de cada método, manejo de excepciones, no-ops cuando inactivo)
- [x] `requirements.txt`: `mlflow>=2.13`; `.env.example`: `MLFLOW_TRACKING_URI=`
- [x] `Makefile`: targets `make train-soft-sensor`, `make train-lstm`, `make mlflow-ui`

### Bloque J — COMPLETADO ✅

Pipeline de reentrenamiento:
- [x] `scripts/retrain.py`: CLI + funciones importables — `load_baseline_metrics`,
      `save_metrics`, `should_promote`, `evaluate_soft_sensor`, `load_train_data`,
      `retrain_soft_sensor`. Métricas base persisten en
      `results/models/purity_soft_sensor.metrics.json`; primer run siempre promueve.
- [x] Criterio de promoción: `new_mae_holdout < baseline * (1 - threshold)`.
      `--force` promueve incondicionalmente. `--threshold 0.02` exige ≥2% mejora.
- [x] Integración MLflow: loguea params, métricas y tag `promoted`.
- [x] `tests/unit/test_retrain.py`: 30 tests (I/O, `should_promote` con 9 casos,
      `evaluate_soft_sensor`, `load_train_data`, integración con sensor mockeado)
- [x] `Makefile`: targets `make retrain`, `make retrain-force`

### Bloque K — COMPLETADO ✅

Validación del simulador contra correlaciones de ingeniería química:
- [x] `simulator/validation.py`: 8 funciones analíticas puras —
      `damkohler_number`, `cstr_da_conversion`, `adiabatic_temperature_rise`,
      `cstr_energy_balance_residual`, `arrhenius_ratio`,
      `vle_y_from_x`, `fenske_minimum_stages`,
      `column_material_balance_error`, `distillate_fraction`.
      Referencias: Fogler §5.3, van Heerden (1953), Fenske (1932), Luyben §8.
- [x] `tests/validation/test_cstr_correlations.py`: 14 tests —
      conversión vs Damköhler (±5%), cota adiabática, cierre de balance
      energético (<15%), sensibilidad Arrhenius (regla de los 10°C).
- [x] `tests/validation/test_column_correlations.py`: 22 tests —
      factibilidad Fenske (N_actual > N_min, margen ≥2 stages), VLE exacta,
      balance de materia, gradiente de temperatura, pureza ≥98.5% en
      operación normal.
- [x] `pytest.ini`: marker `validation` registrado
- [x] `Makefile`: target `make validate-simulator`

### Bloque L — COMPLETADO ✅

Model Status API + History & Models panel en el dashboard:
- [x] `api/server.py`: `GET /api/models/status` — lee
      `results/models/purity_soft_sensor.metrics.json` (escrito por `make retrain`)
      y el directorio del LSTM; devuelve status + métricas + `trained_at` para
      ambos modelos; safe si los archivos no existen
- [x] `tests/unit/test_api_models.py`: 15 tests — sin modelos, con métricas JSON,
      sin métricas (solo joblib), LSTM, ambos entrenados
- [x] `ui/index.html`: tab system en el panel derecho — `OPTIMIZATION` y
      `HISTORY & MODELS` (tabs visibles cuando no hay rec seleccionada)
      - `ModelStatusCard`: muestra estado + métricas del soft sensor y LSTM
      - `ScenarioHistoryTable`: lista escenarios ingestados desde `/api/history/scenarios`
      - `RecentDecisionsTable`: últimas 8 decisiones del operador desde la DB
      - Degradación graciosa: mensajes informativos cuando DB no disponible

### Bloque M — COMPLETADO ✅

Export de reportes HTML:
- [x] `api/report.py`: módulo puro (sin imports de dominio) con 5 funciones:
      `kpi_summary(df)`, `recommendations_summary(recs)`, `decisions_summary(decisions)`,
      `sessions_summary(sessions)`, `render_html(...)` — genera HTML self-contained
      con CSS inline, print-ready via `@media print`, sin dependencias externas.
      Secciones: 4 executive tiles, KPI table (spec violations en rojo),
      recommendations por urgency+rule, recommendations log, decisions log,
      analytics sessions, per-rule performance.
- [x] `api/server.py`: `GET /api/report/current` (HTMLResponse) — convierte
      objetos de dominio a dicts planos, resuelve el status actual de cada
      recomendación (operator_overrides + decisions), llama a las funciones de
      `api/report.py` y retorna el HTML; 404 si no hay escenario cargado
- [x] `tests/unit/test_report.py`: 42 tests — kpi_summary (11), recommendations
      (8), decisions (5), sessions (6), render_html (9), endpoint (3)

### Bloque N — COMPLETADO ✅

Multi-usuario básico (RBAC con tres roles):
- [x] `api/server.py`: middleware `api_key_middleware` extendido con jerarquía
      `viewer < operator < manager`. Permisos por (método, path):
      - GET /api/* → viewer
      - POST /api/recommendations/{id}/decide → operator
      - POST /api/replay/control → operator
      - POST /api/optimization/predict → operator
      - POST /api/scenarios/select y otros writes → manager
- [x] Configuración via env: `AXION_API_KEY_VIEWER`, `AXION_API_KEY_OPERATOR`,
      `AXION_API_KEY_MANAGER`. La clave legacy `AXION_API_KEY` sigue funcionando
      como manager (backward compat). Sin claves seteadas, RBAC se desactiva.
- [x] Respuestas: 401 si la clave es inválida o ausente; 403 con
      `{required_role, actual_role}` si la clave es válida pero el rol no
      alcanza. `/api/health` siempre público.
- [x] Helpers puros expuestos: `_load_role_keys`, `_required_role`,
      `_role_satisfies` — testeables sin levantar el server.
- [x] `tests/unit/test_api_rbac.py`: 31 tests — helpers (9), RBAC desactivado
      (2), viewer (6), operator (4), manager (3), claves inválidas (3),
      legacy compat (2), query param fallback (2)
- [x] `.env.example` actualizado con las 3 nuevas claves de rol

### Bloque O — COMPLETADO ✅

Webhook / notificaciones (Slack/Teams/genérico):
- [x] `api/webhooks.py`: `WebhookNotifier` con dedupe por rec.id, threshold
      configurable (low/medium/high/critical), formato `axion` (estructurado)
      o `slack` (text wrapper), POST async en thread daemon, swallowing de
      errores de red (nunca rompe la request que lo dispara)
- [x] `api/server.py`: integra el notifier en `events_in_window` — cada vez
      que el reloj de replay cruza una recomendación, dispara el webhook si
      `state.webhook.enabled`. Reset del set de dedupe al cambiar escenario.
- [x] `GET /api/webhook/status`: devuelve `{enabled, threshold, format, n_fired}`
      para inspección desde la UI o monitoring externo
- [x] Configuración via env: `AXION_WEBHOOK_URL` (requerida para activar),
      `AXION_WEBHOOK_URGENCY` (default critical), `AXION_WEBHOOK_TIMEOUT`
      (default 5.0s), `AXION_WEBHOOK_FORMAT` (default axion)
- [x] `tests/unit/test_webhooks.py`: 35 tests — threshold (5), payload format
      (4), enabled (2), should_fire (5), notify (8), reset (2), from_env (5),
      endpoint (4)
- [x] Sin dependencia de `requests` ni SDK de Slack/Teams — usa `urllib`
      stdlib para evitar bloat en el container

### Bloque P — COMPLETADO ✅

Escenario custom desde la UI (upload de CSV propio):
- [x] `api/uploads.py`: validador puro `validate_csv_bytes(data: bytes) →
      ValidationResult`. Verifica 20 columnas requeridas (timestamp + 19
      tags de proceso), tamaño (≤25 MB), filas (≥60, ≤100k), parseo de
      timestamp y sanidad numérica por columna (>10% no-numérico → warning,
      100% no-numérico → error). `validate_scenario_name` con regex
      `^[a-z0-9_]{2,40}$`.
- [x] `api/server.py`: `POST /api/scenarios/upload` (multipart/form-data) —
      acepta `file`, `name`, `activate`. Escribe a `data/<name>.csv`,
      activa el escenario si `activate=true`, devuelve 400 con lista de
      errores si la validación falla. Por RBAC requiere rol manager.
- [x] `list_scenarios()` ahora incluye `custom_run` y cualquier upload —
      antes lo filtraba. Reset de `pareto_cache`, `operator_overrides` y
      `webhook` al activar el upload.
- [x] `ui/index.html`: componente `ScenarioUploadButton` en la TopBar — input
      file oculto, prompt para el nombre, POST con FormData, mensajes de
      éxito/error inline (timeout 6s), refresh de la lista de escenarios y
      del panel de recomendaciones tras activar.
- [x] `tests/unit/test_uploads.py`: 25 tests — name validation (8), CSV
      válido (3), errores (8), endpoint (6 — invalid name, missing cols,
      no-activate, with-activate, row count, overwrite)
- [x] `requirements.txt`: agregado `python-multipart>=0.0.9` (requerido por
      FastAPI para multipart/form-data)

### Bloque Q — COMPLETADO ✅

Pipeline LSTM retrain (extiende `scripts/retrain.py`):
- [x] `scripts/retrain.py`: 4 nuevas funciones —
      `aggregate_lstm_metrics(metrics_obj)` aplana el dict
      `by_horizon[h][tgt]` en claves planas (`mae_5min_cstr_T_R_C`, etc.) +
      agregados `mae_overall` (mean) y `mae_worst` (max);
      `should_promote_lstm(new, baseline, threshold)` usa `mae_overall`
      como criterio (vs `mae_holdout` del soft sensor — el LSTM tiene
      time-split interno);
      `load_lstm_train_data(data_dir, scenarios)` lee CSVs y parsea
      timestamps;
      `retrain_lstm(data_dir, lstm_dir, force, threshold, epochs,
      batch_size, val_fraction)` — orquesta todo (lazy import de TF para
      no romper el pipeline del soft sensor cuando TF no está instalado).
- [x] CLI: `--lstm-epochs N` y `--lstm-batch-size N` (defaults 30 / 64);
      `--model lstm` activa solo el LSTM, `--model all` corre ambos en
      secuencia. Cada modelo loguea su propio MLflow Run.
- [x] Persistencia: métricas en `results/models/lstm_forecaster/metrics.json`,
      modelo en el mismo directorio (saved/keras format).
- [x] `tests/unit/test_retrain_lstm.py`: 23 tests — `aggregate_lstm_metrics`
      (6), `should_promote_lstm` (7), `load_lstm_train_data` (4),
      `retrain_lstm` con `_FakeForecaster` mock (6 — sin TF requerido)
- [x] `Makefile`: targets nuevos `make retrain-lstm [FORCE=1]` y
      `make retrain-all [FORCE=1]`

### Bloque R — COMPLETADO ✅

Drift detection (Population Stability Index, PSI):
- [x] `analytics/drift.py`: módulo puro con
      `quantile_bin_edges(values, n_bins=10)` (bins por cuantiles, robusto a
      ties y NaN), `compute_psi(reference, live, edges)` (PSI estándar
      Σ(p_live - p_ref)·ln(p_live/p_ref) con ε para evitar log(0)),
      `classify_psi(psi)` (umbrales Siddiqi: <0.10 none, <0.25 moderate,
      ≥0.25 significant), `DriftDetector(features).fit(ref).score(live)`
      → `DriftReport` con per-feature PSI + overall worst.
- [x] `api/server.py`: `state.drift_detector` se entrena en startup sobre el
      mismo `normal.csv` que el AnalyticalEngine (ref del soft sensor).
      `GET /api/drift/status?window=240` devuelve report sobre los últimos
      N samples (default 240 = 4h a 1 sample/min); `warming_up` si <30
      muestras; 404 si no hay escenario.
- [x] `ui/index.html`: componente `DriftBadge` que polea cada 30s y muestra
      verde (IN-DOMAIN), ámbar (MODERATE) o rojo (SIGNIFICANT) en la card
      del soft sensor; tooltip con el feature peor + PSI value.
- [x] `tests/unit/test_drift.py`: 29 tests — `classify_psi` (6),
      `quantile_bin_edges` (5), `compute_psi` (6 incluyendo identidad ≈ 0,
      shift de 2σ → PSI > 0.5, NaN handling), `DriftDetector` (8),
      endpoint `/api/drift/status` (4)

### Bloque S — COMPLETADO ✅

Outcome tracking real (cierra el loop sobre decisiones via UI):
- [x] `consensus/operator_outcomes.py`: módulo puro con
      `OperatorOverride` (rec_id, status, justification, decision_ts),
      `is_outcome_measurable(override, rec, now_ts)` (true sólo si
      accepted/modified y la ventana de medición ya pasó),
      `synthesize_decision_and_execution(rec, override)` (construye
      Decision/Execution con `executor=operator`),
      `measure_one(override, rec, df, tracker)` reutiliza el
      `OutcomeTracker` existente,
      `measure_pending(...)` itera todos los overrides y devuelve sólo los
      outcomes nuevos (idempotente — caller pasa el set de ya medidos),
      `outcome_summary_dict(outcome, rec)` para el wire format.
- [x] `api/server.py`: cuando el operador decide, ahora se persiste
      `decision_ts` (timestamp simulado del replay). En cada tick del
      replay loop se llama `_measure_due_operator_outcomes(now_ts)` que
      mide los outcomes recién maduros y alimenta el
      `PerformanceTracker` (record_outcome) para que la confianza por
      regla refleje las decisiones del usuario.
- [x] `GET /api/outcomes/operator?limit=N` — devuelve outcomes con
      predicted_value vs actual_value por variable, deviation_pct,
      within_tolerance, quality_score.
- [x] `ui/index.html`: nuevo componente `OperatorOutcomesTable` en la
      tab HISTORY & MODELS — polea cada 15s, semáforo verde/ámbar/rojo
      según quality_score (≥0.75 / ≥0.50 / <0.50), columnas: regla,
      quality, delay, measured timestamp.
- [x] `tests/unit/test_operator_outcomes.py`: 24 tests —
      `is_outcome_measurable` (6), `synthesize` (4), `measure_one` (4),
      `measure_pending` (5), `outcome_summary_dict` (2), endpoint (3)

### Bloque T — COMPLETADO ✅

OPC-UA real (PLC / DCS / Kepware / Prosys):
- [x] `integration/integration_service.py`: orquestador con
      `IntegrationStatus` (enabled, connected, endpoint, samples_received,
      last_sample_ts, last_error, n_tags, started_at) y
      `IntegrationService.from_env()` que devuelve `None` cuando
      `AXION_OPCUA_ENABLED` no está seteada (no se levanta el cliente).
      Lifecycle `start()` / `stop()` respetuoso (timeout 3s al cerrar).
- [x] `integration/opcua_source.py`: agregado soporte de `security_policy`
      (Basic256Sha256) via `client.set_security_string(...)` cuando se
      configuran `cert_path` + `key_path`. Backward compatible — `"None"`
      sigue siendo el default.
- [x] `integration/tag_map.py`: `ServerConfig` extendido con `cert_path`,
      `key_path`, `security_mode` (default `SignAndEncrypt`).
- [x] `load_tag_map_from_env()` resuelve la prioridad: archivo
      (`AXION_OPCUA_TAG_MAP`) → default in-memory `PILOT_TAG_MAP`. Aplica
      overrides de env (endpoint, username, password, security, certs)
      por encima del map elegido.
- [x] `api/server.py`: startup levanta `IntegrationService.from_env()` si
      está configurado; shutdown lo detiene gracefully. Endpoint
      `GET /api/integration/opcua/status` siempre responde 200 con el
      snapshot de salud (la UI lo polea cada 5s).
- [x] `ui/index.html`: badge `OpcuaBadge` en la TopBar — solo visible
      cuando el servicio está habilitado, verde con contador de samples
      cuando connected, rojo "OFFLINE" con tooltip de último error.
- [x] `.env.example`: 8 nuevas variables `AXION_OPCUA_*`
- [x] `tests/unit/test_integration_service.py`: 33 tests — `_is_truthy`
      (10 con parametrize), `_override_from_env` (4), `load_tag_map_from_env`
      (3 incluyendo override file→env), `IntegrationStatus` (2), `from_env`
      (2), callbacks (5 incluyendo on_sample exception handling),
      lifecycle con OPCUASource mockeado (2), endpoint (2)

### Bloque T' — COMPLETADO ✅

OPC-UA → state.run (cierra el loop de Bloque T):
- [x] `api/data_source.py`: `OpcuaBuffer` thread-safe (deque con cap 14_400 ≈ 4h
      a 1 sample/seg), `to_dataframe()` materializa el buffer en el shape
      canónico `LIVE_COLUMNS` (proyecta tags conocidos, NaN-fill para los
      ausentes, drop de tags desconocidos). Idempotente y reutilizable.
- [x] `api/server.py`: `state.opcua_buffer` y `state.data_source` ∈
      {"replay","opcua"} (default "replay"). El callback `_on_opcua_sample`
      del `IntegrationService` empuja samples al buffer. El replay loop
      llama `_refresh_live_run()` cuando `data_source == "opcua"` —
      reconstruye `state.run` con `scenario="live"` y re-corre AnalyticalEngine
      + RecommendationEngine cada ≥30 muestras nuevas (cheap path: solo
      avanza el cursor).
- [x] Endpoints nuevos: `POST /api/data-source/select` (body `{"source":
      "replay"|"opcua"}`, 503 si OPC-UA no configurado, limpia overrides /
      outcomes / webhook dedupe, broadcast WebSocket
      `data_source_changed`); `GET /api/data-source/status` con
      `{source, opcua_enabled, opcua_connected, buffer_samples}`.
      `GET /api/replay/status` ahora incluye `data_source`.
- [x] `ui/index.html`: el badge `OpcuaBadge` ahora es un botón clickeable —
      toggle `replay` ↔ `opcua` con confirmación del backend; muestra
      "OPC-UA · {N} · LIVE" cuando el stream está activo.
- [x] `tests/unit/test_data_source.py`: 18 tests — `OpcuaBuffer` (10 incluye
      capacity rolling, NaN fill, drop de tags desconocidos, timestamp
      inválido), endpoint select (4 incluye 400 inválido, 503 sin OPC-UA,
      switch a opcua, switch a replay con load_scenario stub), endpoint
      status (2), replay/status expone source (2)

### Bloque U — COMPLETADO ✅ (slice 1: profile abstraction + segundo proceso)

Multi-proceso (la plataforma deja de ser one-trick-pony):
- [x] `profile/process_profile.py`: `ProcessProfile` declarativo con `TagSpec`
      (tag, label, units, spec_min/max, is_kpi), feature_cols, target_col,
      scenarios, purity_kpi (+ spec_min). Registry global con `register`,
      `get_profile`, `list_profiles`, `active_profile()` que respeta
      `AXION_PROCESS_PROFILE` env var (default "pilot").
- [x] `profile/profiles.py`: dos perfiles concretos —
      `PILOT_PROFILE` (CSTR + columna, 19 tags, 9 escenarios, purity_kpi
      `column.purity_B` ≥98.5%); `BATCH_PROFILE` (reactor batch exotérmico,
      8 tags `batch.*`, 3 escenarios, purity_kpi `batch.conversion` ≥0.85).
      Namespaces disjuntos para que ambos coexistan sin colisiones.
- [x] `simulator/batch_reactor.py`: ODE solver (scipy RK45) — A→P, balance
      energético reactor + jacket, kinetics Arrhenius. `simulate_batch(params,
      start_time)` produce DataFrame en shape canónico. `BatchParams`
      acepta schedules step-function para `F_cool` y `T_cool_in` (sirven
      para construir runaway scenarios).
- [x] `scripts/generate_batch_scenarios.py`: produce
      `data/{batch_normal, batch_runaway, batch_slow_kinetics}.csv` (241
      muestras / 4h / 1 sample/min). Target Makefile
      `make generate-batch-scenarios`.
- [x] `api/report.py`: `kpi_summary(df, profile=None)` y `render_html(...,
      profile=None)` ahora son profile-aware. `_kpi_defs(profile)` construye
      la lista de KPIs desde los `kpi_tags` del profile en vez de un dict
      hardcodeado. El tile ejecutivo usa `profile.purity_kpi` y su label/units.
- [x] `api/data_source.py`: `LIVE_COLUMNS` ahora es la concatenación
      `["timestamp"] + active_profile().tag_names` via `get_live_columns()`.
      `OpcuaBuffer` consulta el profile al proyectar samples.
- [x] `api/server.py`: endpoints `GET /api/profile` (active + available +
      profile dict completo) y `POST /api/profile/select` con body
      `{"profile": "pilot"|"batch_reactor"}` (400 si nombre inválido,
      setea env var, broadcast WebSocket `profile_changed`).
- [x] `tests/unit/test_profile.py`: 31 tests — schema (5),
      registry (6 incluyendo env override), perfiles concretos (5 incluyendo
      namespace disjoint), simulador batch (6 incluyendo runaway > normal,
      conversion monotónica), report profile-aware (4 incluyendo
      "no leakage" pilot↔batch), endpoints (5)
- [x] `.env.example`: `AXION_PROCESS_PROFILE=pilot`

### Bloque U2 — COMPLETADO ✅

Rule packs por profile + analytics profile-aware:
- [x] `profile/process_profile.py`: `ProcessProfile` extendido con
      `measured_tags` (subset del tag list — los que el FrozenSensorDetector
      observa), `operational_limits` (dict por tag con `low`/`high`/
      `rate_per_min` para el TrendDetector), y `rule_pack_path`
      (string `"module:attribute"` que se resuelve via `importlib` en
      `load_rules()`, lazy para no arrastrar pandas/recs al importar
      profile).
- [x] `profile/profiles.py`: ambos perfiles ahora declaran
      `measured_tags`, `operational_limits` y `rule_pack_path`.
      `PILOT_PROFILE` apunta a `recommendations.rules_pilot:PILOT_RULES`,
      `BATCH_PROFILE` a `recommendations.rules_batch:BATCH_RULES`.
- [x] `recommendations/rules_batch.py`: starter rule pack con 3 reglas —
      `B01_HighReactorTemp` (T_R sobre spec → propone abrir coolant),
      `B02_RunawayRisk` (CRITICAL si dHdt + T_R co-ocurren → max coolant +
      flag de safety review), `B03_LowConversion` (informativa,
      INVESTIGATE action sin predicción numérica). Misma API
      `DiagnosticRule` que el pilot pack — el `RecommendationEngine`,
      `ConsensusController` y `OutcomeTracker` funcionan sin cambios.
- [x] `api/server.py`: startup ahora construye `AnalyticalEngine` y
      `RecommendationEngine` con los `tags` / `operational_limits` /
      `rules` del active profile. Helper `_baseline_csv_for(profile)`
      elige automáticamente `<profile>_normal.csv` o el primer escenario
      declarado. `POST /api/profile/select` rebuilda los engines (+ drift
      detector + recarga primer escenario disponible) contra el nuevo
      profile, con fallback graceful si algo falla (mantiene los engines
      anteriores y devuelve `warning` en la respuesta).
- [x] `tests/unit/test_rules_batch.py`: 15 tests — B01 (5), B02 (4
      incluye dependencia de co-occurring T_R), B03 (3 incluye action
      INVESTIGATE sin numeric impact), pack (3 — count, names únicos,
      prefijo B).
- [x] `tests/unit/test_profile_engines.py`: 13 tests — `load_rules` (3),
      schema adds (5 incluye `measured_tags ⊆ tags`), AnalyticalEngine
      por profile (3 — pilot fits pilot data, batch fits batch data, no
      cross-leakage), RecommendationEngine por profile (2)

### Bloque V — COMPLETADO ✅

RBAC completo + JWT (auth con tokens y usuarios en DB):
- [x] `api/auth.py`: módulo puro con `hash_password` / `verify_password`
      (bcrypt), `encode_token` / `decode_token` (PyJWT HS256), `TokenClaims`
      (dataclass con sub/uid/role/type/iat/exp), `issue_token_pair` que
      devuelve (access, refresh, expires_in). Errores tipados —
      `AuthError`, `InvalidTokenError`, `TokenExpiredError`,
      `TokenTypeMismatchError` — para que el server mapee a 401.
      `extract_bearer_token(header)` parsea `Authorization: Bearer ...`
      tolerando case y espacios.
- [x] `db/versions/002_users.py`: migración Alembic — tabla `users`
      (id, email UNIQUE, password_hash, role CHECK
      `('viewer','operator','manager')`, active, created_at) +
      `idx_users_email`.
- [x] `db/users.py`: `UserRecord` dataclass + `UserRepository` con
      `get_by_email`, `get_by_id`, `list_all`, `create`, `update_role`,
      `update_password`, `set_active`, `delete`. `to_public_dict()` no
      expone el password_hash.
- [x] `scripts/users.py`: CLI con subcomandos `create`, `list`, `set-role`,
      `reset-password`, `deactivate`. Pide password con `getpass` (con
      confirmación), exit codes específicos por error. Target Makefile:
      `make users CMD="create --email a@b.com --role manager"`.
- [x] `api/server.py`: middleware extendido —
      orden de resolución `Authorization: Bearer` → API key fallback. Si
      JWT presente, valida firma + expiración + tipo (access). El JWT
      tiene precedencia: una key API válida no levanta el rol si el JWT
      es viewer. 401 con `{"code": "token_expired"}` cuando la access
      caducó (señal explícita para que el cliente refresque). Paths
      siempre públicos: `/api/health`, `/api/auth/login`,
      `/api/auth/refresh`.
- [x] Endpoints nuevos:
      - `POST /api/auth/login` (body `{email, password}`) → access +
        refresh + user dict. 503 si falta secret/DB; 401 credenciales
        inválidas o usuario inactivo.
      - `POST /api/auth/refresh` (body `{refresh_token}`) → nuevo par
        de tokens. 401 si refresh expiró/inválido o si el usuario fue
        desactivado mientras tanto. Rechaza tokens type=access.
      - `GET /api/auth/me` → `{email, uid, role}` desde el state del
        request (poblado por el middleware).
- [x] `requirements.txt`: `PyJWT>=2.8`, `bcrypt>=4.0`. `.env.example`:
      `AXION_JWT_SECRET`, `AXION_JWT_ACCESS_MINUTES` (default 30),
      `AXION_JWT_REFRESH_DAYS` (default 7).
- [x] `tests/unit/test_auth_jwt.py`: 41 tests — password hashing (6),
      token encode/decode (7 incluye expired, wrong secret, type
      mismatch, missing claim), `issue_token_pair` (5),
      `extract_bearer_token` (5), endpoint `/api/auth/login` (7 incluye
      503 sin secret/DB, 400 sin email, 401 user-not-found,
      401 bad password, 200 success, inactive rejected),
      `/api/auth/refresh` (4), `/api/auth/me` (2), middleware con JWT
      (5 incluye precedencia sobre API key, expired→401 con código,
      `/api/auth/login` siempre público).

### Bloque W — COMPLETADO ✅

Rate limiting + observabilidad (Prometheus):
- [x] `api/metrics.py`: registry dedicado `REGISTRY` con 7 métricas —
      `axion_http_requests_total` (counter, labels method/path/status),
      `axion_http_request_duration_seconds` (histogram con buckets
      5ms-10s), `axion_http_inflight_requests` (gauge),
      `axion_recommendations_total` (counter rule/urgency/scenario),
      `axion_operator_decisions_total` (counter por status),
      `axion_websocket_connections` (gauge),
      `axion_rate_limit_rejections_total` (counter por limiter type).
      `template_path()` colapsa paths dinámicos (REC-1234, scenario
      names) a `{rec_id}` / `{name}` para evitar cardinality blowup.
      `RequestTimer` context manager incrementa inflight + observa duración.
- [x] `api/rate_limit.py`: token-bucket thread-safe `RateLimiter` con
      `from_env()`, `allow(identity) → (allowed, retry_after_s)`,
      `reset()`. `resolve_identity()` con prioridad
      user_id > api_key (sha8) > client_ip > "anonymous" — usuario
      logueado tiene su propio bucket aunque comparta IP NAT'd.
      Configurable via `AXION_RATE_LIMIT_PER_MIN` (default 120, 0
      desactiva), `AXION_RATE_LIMIT_BURST`. Exempt paths: `/api/health`,
      `/api/metrics` (nunca rate-limited).
- [x] `api/server.py`: middleware `metrics_and_rate_limit` registrado
      DESPUÉS del auth middleware (LIFO en FastAPI → corre antes en
      ejecución reversed). 429 con `{retry_after}` body + `Retry-After`
      header cuando se excede. `state.rate_limiter` reemplazable en
      tests. Hooks específicos:
      `operator_decisions_total.labels(status).inc()` cuando el operador
      decide; `recommendations_total.labels(rule, urgency, scenario).inc()`
      cuando una rec cruza el reloj de replay;
      `websocket_connections.set(len(state.clients))` en accept/disconnect.
- [x] Endpoint `GET /api/metrics`: Prometheus text format vía
      `render_prometheus()`. Exento de auth y rate-limit (scrapers pueden
      poll sin credenciales — métricas no contienen PII ni secretos).
      Path agregado a `_AUTH_PUBLIC_PATHS`.
- [x] `requirements.txt`: `prometheus-client>=0.20`. `.env.example`:
      `AXION_RATE_LIMIT_PER_MIN`, `AXION_RATE_LIMIT_BURST`.
- [x] `tests/unit/test_rate_limit.py`: 21 tests — `resolve_identity`
      priority chain (5), `RateLimiter` (9 incluye disabled-when-zero,
      first burst allowed, replenishment con monkeypatched monotonic,
      retry_after estimation, reset specific/all), `from_env` (5).
- [x] `tests/unit/test_metrics.py`: 18 tests — `template_path` (4),
      `render_prometheus` shape (2), `RequestTimer` inflight/duration
      side effects (2), endpoint `/api/metrics` (4 incluye exempt de
      JWT), middleware request recording (2),
      rate-limit middleware integrado (4 incluye 429 después del burst,
      health/metrics exempt).

### Bloque X — COMPLETADO ✅

Tests E2E + smoke con TimescaleDB en CI:
- [x] `tests/e2e/conftest.py`: fixture session-scoped `live_server` que
      spawnea un uvicorn subprocess en un puerto libre, hace polling de
      `/api/health` hasta 15s y entrega la base URL. Auth/RL/DB/OPC-UA
      desactivados en el subprocess para que la suite sea hermética.
      Hook `pytest_collection_modifyitems` skipea tests con marker
      `browser` salvo que se pase `--run-browser`. Fixture `http_client`
      con timeout 30s para los listings pesados de recommendations.
- [x] `tests/e2e/test_api_e2e.py`: 16 tests E2E (httpx contra el server
      real) — health (2), scenarios (3 incluye 404), state/recent (2),
      replay status (1), decision round-trip (1, hace seek + accept +
      verify), profile (2), metrics (1), models/drift (2), report (1
      verifica HTML real), websocket lifecycle (1, conecta y recibe
      snapshot inicial). Catches el wiring entre middleware/startup
      hooks/replay loop que el unit-level TestClient no ejercita.
- [x] `tests/e2e/test_dashboard_e2e.py`: 3 tests Playwright (browser)
      gateados por `--run-browser`. Importa `playwright.sync_api`
      lazy → si no está instalado, skip graceful. Verifica brand
      render, scenario picker presente, y round-trip de scenario change.
- [x] `pytest.ini`: markers `e2e` y `browser` registrados.
      `Makefile`: `make test-e2e` (httpx) y `make test-e2e-browser`
      (Playwright opt-in).
- [x] `.github/workflows/ci.yml`: nuevo job `e2e` con
      `services.timescaledb` (timescale/timescaledb:latest-pg16,
      health-check), `AXION_DB_URL` apuntando al servicio, aplica
      migraciones Alembic (incluye 002_users de Bloque V) y corre
      `pytest tests/e2e/ -m e2e`. Browser tests NO se corren en CI —
      son opt-in local. Total CI jobs: `fast` (PR + push) +
      `slow` (push) + `e2e` (PR + push).

### Bloque Y — COMPLETADO ✅

Containerización completa (un `docker compose up` y listo):
- [x] `Dockerfile`: multi-stage build (Python 3.11-slim) — stage builder
      instala build-essential + libpq-dev, crea venv en `/opt/venv` con
      `requirements.txt`. Stage runtime copia el venv, mantiene solo
      libpq5 + curl, crea usuario no-root `axion` (uid 1000), drop a
      USER axion, EXPOSE 8000, HEALTHCHECK con curl contra `/api/health`.
      Entrypoint: `uvicorn api.server:app --host 0.0.0.0 --port 8000`.
- [x] `.dockerignore`: excluye `.git`, `__pycache__`, venvs, `.env`,
      `mlruns`, `tests/`, etc. para imágenes lean.
- [x] `docker-compose.yml`: 4 servicios — `timescaledb` (existente),
      `migrate` (one-shot, `alembic upgrade head`, gating de
      `axion-api`), `mlflow` (ghcr.io/mlflow/mlflow:v2.16.2 con SQLite
      backend en volumen `mlruns`, healthcheck), `axion-api`
      (build local, depends_on con conditions service_healthy /
      service_completed_successfully / service_started, env vars
      cableados — DB URL apunta a `timescaledb`, MLflow apunta a
      `mlflow:5000`). Healthchecks en todos los servicios.
- [x] `Makefile`: targets `make stack-build`, `make stack-up`
      (build + up con polling de healthy hasta 60s), `make stack-logs`
      (tail), `make stack-down` (preserva volúmenes), `make stack-reset`
      (down -v).
- [x] `DEMO.md`: sección "Stack completo containerizado" con los 4
      comandos y los 3 puertos expuestos (8000 dashboard, 5000 MLflow,
      5432 DB).
- [x] `tests/unit/test_compose_config.py`: 19 tests sin Docker
      requerido — parsea YAML y verifica la forma:
      `TestComposeStructure` (9 — servicios presentes, depends_on,
      DB URL apunta al service, MLflow URI, port 8000 publicado,
      volúmenes, healthcheck), `TestDockerfile` (5 — multistage,
      USER axion, EXPOSE 8000, HEALTHCHECK con /api/health,
      uvicorn entrypoint), `TestDockerignore` (5).

### Bloque Z — COMPLETADO ✅

Versioning git-like de los CSVs de training:
- [x] `data_versioning/snapshots.py`: módulo puro con
      `FileMeta` (name, sha256, size_bytes, n_rows, n_cols, modified_at,
      relpath) y `Snapshot` (snapshot_id, created_at, message, files).
      `compute_file_meta` hashea con SHA256 streaming (chunks de 64 KB),
      cuenta filas/columnas sin cargar el CSV completo. `compute_snapshot_id`
      es content-addressed: id = `sha256("\\n".join(f"{name}={sha}"
      sorted)).hexdigest()[:12]` — el mismo contenido siempre produce el
      mismo id (deduplicación gratuita). `take_snapshot`, `load_snapshot`,
      `list_snapshots` (orden cronológico), `diff_snapshots`
      (added/removed/changed) y `verify_snapshot`
      (ok/missing/extra/mismatched) cierran el set.
- [x] `scripts/version_data.py`: CLI con subcomandos `snapshot`,
      `list`, `show <id>`, `diff <a> <b>`, `verify <id>` (exit code != 0
      si diverge). Output formateado: tabla con id/created/files/message.
- [x] `scripts/retrain.py`: tras cada train (soft sensor + LSTM) toma
      automáticamente un snapshot de `data/` y guarda
      `data_snapshot_id` + `data_snapshot_files` en `metrics.json`.
      Cualquier modelo persistido a partir de ahora deja trazabilidad
      completa de qué datos vio.
- [x] `api/server.py`: 3 endpoints nuevos —
      `GET /api/data/snapshots` (lista con id/created/message/n_files),
      `GET /api/data/snapshots/{id}` (manifest completo, 404 si no existe),
      `GET /api/data/snapshots/{id}/verify` (re-hashea data/, devuelve
      `{ok, missing, extra, mismatched}`).
- [x] `Makefile`: `make data-snapshot [MSG="..."]`,
      `make data-snapshots`, `make data-verify ID=<id>`.
- [x] `tests/unit/test_data_versioning.py`: 33 tests —
      `compute_file_meta` (6 incluye n_rows excludes header,
      unparseable handling), `compute_snapshot_id` (4 incluye determinism
      e independence-of-message), snapshot I/O (6 incluye round-trip JSON,
      raise on empty dir, list empty), `diff_snapshots` (4 — no diff,
      added, removed, changed), `verify_snapshot` (4 — ok, modified,
      missing, extra), Snapshot dataclass (2), endpoints
      `/api/data/snapshots*` (7 — empty list, list after take,
      get manifest, 404, verify ok, verify modified, verify 404).

## Convenciones del proyecto

- **Idioma**: código y comentarios en inglés. Docstrings en inglés.
  Comunicación con el desarrollador en español.
- **Tests**: cualquier código nuevo de dominio requiere tests unitarios.
  Coverage objetivo: >75% en módulos de dominio, >50% en glue code.
- **Logs en producción**: nunca `print()`. Usar `axion_logging.get_logger()`.
- **Determinismo**: cualquier test que use random fija seed.

## Comandos útiles

```bash
# Tests
make test                                  # fast suite (~15s)
make test-slow                             # con LSTM/NSGA-II (~30s)
make coverage                              # HTML report → results/coverage_html/
make smoke                                 # arranca server, golpea endpoints, para

# Servidor
make serve                                 # uvicorn --reload --port 8000

# Simulador standalone
python run_simulation.py --scenario thermal_drift

# Base de datos (requiere Docker)
docker compose up -d                       # arranca TimescaleDB
cp .env.example .env                       # configurar credenciales
AXION_DB_URL=postgresql://axion:axion@localhost:5432/axion \
  python -m alembic -c db/alembic.ini upgrade head   # aplicar migraciones
AXION_DB_URL=postgresql://axion:axion@localhost:5432/axion \
  python scripts/ingest_csvs.py           # ingestar todos los CSVs
AXION_DB_URL=... python scripts/ingest_csvs.py --force normal  # re-ingestar uno

# ML training + experiment tracking
make train-soft-sensor                     # entrena GBR purity sensor → results/models/
make train-lstm                            # entrena LSTM forecaster (requiere TF)
make mlflow-ui                             # abre UI en http://localhost:5000
make retrain                               # reentrenar + promover si MAE mejoró
make retrain-force                         # reentrenar + promover siempre
python scripts/retrain.py --threshold 0.02 # exigir ≥2% mejora para promover
make validate-simulator                    # checks primera ley vs CSVs reales
```

## Prioridad estratégica actual

**Validación de demanda > desarrollo técnico**. El equipo decidió
explícitamente pausar hardening para enfocarse en entrevistas con ICP
y aplicaciones a incubadoras. Cualquier trabajo técnico debe justificarse
en términos de qué desbloquea para esa validación (ej: smoke test sólido
para que un evaluador de incubadora pueda correr la demo sin fricción).
