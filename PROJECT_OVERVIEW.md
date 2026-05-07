# Axion AI — Overview del proyecto

> Documento de lectura. Resumen completo de qué es Axion AI hoy, qué tiene
> construido, y qué decisiones quedan pendientes para el próximo bloque
> de trabajo.

---

## 1. Qué es

Axion AI es una plataforma de **ingeniería de procesos autónoma** para
plantas industriales. La pregunta que responde, en voz de un plant
manager, es:

> "¿Qué está pasando ahora en mi proceso, qué va a pasar en 30 minutos,
> y si tengo que actuar — qué acción concreta tomo?"

El stack:

- **Simulador** (CSTR + columna de destilación + reactor batch) que
  produce datos sintéticos cuando no hay planta real conectada.
- **Analítica** (SPC, PCA, Trend, Regime, FrozenSensor) que detecta
  anomalías sobre los streams de tags.
- **Soft sensor** (GBR ensemble) que predice variables que no se miden
  directamente — pureza del producto en el caso del pilot.
- **Forecaster LSTM** multi-target multi-horizon (5/15/30/60 min).
- **Recommendation engine** con reglas R1–R10 (pilot) + B01–B03 (batch),
  cada una con diagnóstico, causa probable, acción y predicción de impacto.
- **Optimizer NSGA-II** que devuelve el frente de Pareto del proceso
  para 4 objetivos (pureza, energía, producción, estabilidad).
- **Consensus controller** con 3 modos (ADVISOR, SEMI-AUTONOMOUS,
  AUTONOMOUS) y safety gate.
- **API FastAPI + WebSocket** que sirve todo eso a un dashboard React.

**Estado actual: v0.9, demo-ready, no production-ready** (todavía no
soak-tested contra una planta real).

---

## 2. Pilot vs. Multi-proceso

La plataforma soporta **dos perfiles de proceso** out of the box:

| Profile        | Display name                  | Tags  | Reglas | Escenarios CSV  |
| -------------- | ----------------------------- | ----- | ------ | --------------- |
| `pilot`        | CSTR + Distillation Column    | 19    | R1-R10 | 9               |
| `batch_reactor`| Exothermic Batch Reactor      | 8     | B1-B3  | 3               |

Switch en runtime: `POST /api/profile/select {"profile": "..."}`. El
servidor reconstruye `AnalyticalEngine`, `RecommendationEngine` y el
`DriftDetector` contra el nuevo perfil. La activación es graceful — si
algo falla, los engines anteriores quedan en pie y la respuesta lleva
un `warning`.

**Adaptarlo a un tercer proceso requiere:**

1. Declarar un nuevo `ProcessProfile` con sus tags/KPIs/escenarios
2. Escribir un rule pack (3-10 reglas en estilo `B01_*`)
3. Producir CSVs de escenario (idealmente desde un simulador del nuevo
   proceso, o pegando datos reales del cliente)
4. Opcional: definir `operational_limits` para el TrendDetector

No hay cambios de schema de DB necesarios — los samples viven en una
tabla wide pero las columnas no existentes simplemente quedan NULL.

---

## 3. Bloques A–Z entregados

Resumen de cada bloque, en orden cronológico:

| Bloque | Nombre                          | Punto técnico clave                                                                          |
| ------ | ------------------------------- | -------------------------------------------------------------------------------------------- |
| A      | Test suite                      | 269 tests unitarios + pytest.ini + fixtures                                                  |
| B      | Demo hardening                  | `axion_logging` JSON estructurado; `Makefile`; `smoke_test.sh`                               |
| C      | Persistencia TimescaleDB        | docker-compose, Alembic, `process_samples` hypertable, ingest_csvs.py                        |
| D      | DB integrada al server          | `DbClient` (write), startup connect graceful, persiste samples + recs + decisions            |
| E      | `make demo`                     | Comando único que levanta DB + migrate + serve                                               |
| F      | CI/CD                           | GitHub Actions: fast (PR+push, 3.11+3.12), slow (push)                                       |
| G      | Auth API key                    | Middleware X-API-Key, query param para WS, desactivable                                      |
| H      | Endpoints históricos            | `GET /api/history/{scenarios,samples,recommendations,decisions}` con filtros                 |
| I      | MLflow tracking                 | Wrapper `Run` con no-ops graciosos, integrado en ambos training scripts                      |
| J      | Pipeline reentrenamiento        | `scripts/retrain.py` con criterio de promoción `mae_holdout < baseline*(1-threshold)`        |
| K      | Validación simulador            | 9 funciones puras (Damköhler, Fenske, Arrhenius, VLE), 36 tests vs CSVs reales               |
| L      | Model status + History UI       | `GET /api/models/status`, tab "HISTORY & MODELS" en dashboard                                |
| M      | Reportes HTML                   | `api/report.py` puro, `GET /api/report/current` self-contained printable                     |
| N      | RBAC con 3 roles                | viewer/operator/manager via API key; jerarquía explícita                                     |
| O      | Webhooks                        | POST async stdlib (urllib), threshold configurable, dedupe por rec_id, formatos axion/slack  |
| P      | Upload de CSV                   | `POST /api/scenarios/upload` con validador puro, integrado en TopBar de la UI                |
| Q      | LSTM retrain pipeline           | `aggregate_lstm_metrics`, `should_promote_lstm`, lazy import de TF                           |
| R      | Drift detection (PSI)           | Population Stability Index, umbrales Siddiqi, badge en `ModelStatusCard`                     |
| S      | Outcome tracking real           | Decisión via UI → schedule de medición → `OutcomeTracker` reutilizado, alimenta performance  |
| T      | OPC-UA real                     | `IntegrationService` con security policy + status endpoint + badge en TopBar                 |
| T'     | OPC-UA → state.run              | `OpcuaBuffer` thread-safe, switch replay↔opcua via `POST /api/data-source/select`            |
| U      | Multi-proceso (profile)         | `ProcessProfile` + `BATCH_PROFILE` + simulador batch reactor (scipy RK45) + 3 escenarios     |
| U2     | Rule packs por profile          | `rule_pack_path` lazy resolver, B01-B03, AnalyticalEngine consume profile.tags + limits      |
| V      | RBAC completo + JWT             | PyJWT HS256 + bcrypt + `users` table + `/api/auth/{login,refresh,me}`                        |
| W      | Rate limiting + Prometheus      | Token-bucket por identidad, 7 métricas, `/api/metrics` exempt de auth                        |
| X      | Tests E2E + CI con DB           | `tests/e2e/` con uvicorn subprocess + Playwright opt-in + CI job con TimescaleDB             |
| Y      | Containerización                | Dockerfile multi-stage non-root, docker-compose con 4 servicios, `make stack-up`             |
| Z      | Versioning de escenarios        | Snapshots content-addressed (SHA256, 12-hex id), auto-stamp en metrics.json                  |

---

## 4. Estructura del repo (al día de hoy)

```
axion_simulator/
├── api/
│   ├── server.py              FastAPI app — todos los endpoints
│   ├── auth.py                JWT + bcrypt helpers
│   ├── rate_limit.py          Token-bucket limiter
│   ├── metrics.py             Prometheus registry + metrics
│   ├── webhooks.py            WebhookNotifier
│   ├── report.py              HTML report generator (puro)
│   ├── uploads.py             CSV validator (puro)
│   └── data_source.py         OpcuaBuffer + LIVE_COLUMNS profile-aware
├── analytics/
│   ├── engine.py              AnalyticalEngine (orquestador)
│   ├── spc.py, pca.py, trend.py, regime.py, frozen.py
│   ├── sessions.py            Agrupa alerts en EventSession
│   └── drift.py               PSI / DriftDetector (Bloque R)
├── recommendations/
│   ├── engine.py              RecommendationEngine
│   ├── rules_base.py          DiagnosticRule abstract + RuleContext
│   ├── rules_pilot.py         R01-R10 (pilot)
│   └── rules_batch.py         B01-B03 (batch)
├── consensus/
│   ├── controller.py          ConsensusController
│   ├── outcome.py             OutcomeTracker + PerformanceTracker
│   └── operator_outcomes.py   Closes the loop sobre UI decisions
├── soft_sensor/               GBR ensemble para purity_B
├── predictive/                LSTM forecaster (TF, lazy-loaded)
├── optimizer/                 NSGA-II + ProcessSurrogate
├── simulator/
│   ├── cstr.py, column.py     ODEs del pilot
│   ├── batch_reactor.py       ODEs del batch (Bloque U)
│   ├── scenarios.py           Scripts que generan CSVs del pilot
│   └── validation.py          9 funciones analíticas (Bloque K)
├── integration/
│   ├── opcua_source.py        Cliente OPC-UA
│   ├── opcua_writer.py        Setpoint writes con safety gates
│   ├── opcua_mock_server.py   Mock server local para desarrollo
│   ├── tag_map.py             Declarative mapping
│   ├── integration_service.py Lifecycle wrapper (Bloque T)
│   └── ingestion.py
├── profile/
│   ├── process_profile.py     ProcessProfile + TagSpec + registry
│   └── profiles.py            PILOT_PROFILE + BATCH_PROFILE
├── data_versioning/
│   └── snapshots.py           Content-addressed snapshots (Bloque Z)
├── db/
│   ├── client.py              DbClient (psycopg2)
│   ├── users.py               UserRepository (Bloque V)
│   ├── alembic.ini, env.py
│   └── versions/
│       ├── 001_initial.py
│       └── 002_users.py
├── scripts/
│   ├── ingest_csvs.py
│   ├── smoke_test.sh
│   ├── retrain.py             CLI de retrain (soft sensor + lstm)
│   ├── users.py               CLI de user management (Bloque V)
│   ├── version_data.py        CLI de snapshots (Bloque Z)
│   └── generate_batch_scenarios.py
├── examples/
│   ├── train_soft_sensor.py
│   └── train_lstm_forecaster.py
├── ui/index.html              React standalone (no build step)
├── data/                      Escenarios CSV
│   ├── normal.csv, thermal_drift.csv, ... (pilot)
│   ├── batch_normal.csv, batch_runaway.csv, batch_slow_kinetics.csv
│   └── .versions/<id>.json    Manifests de snapshots
├── results/
│   └── models/                Modelos entrenados + metrics.json
├── tests/
│   ├── unit/                  ~735 tests
│   ├── e2e/                   16 tests httpx + 3 Playwright (opt-in)
│   └── validation/            36 tests vs first-principles
├── .github/workflows/ci.yml   3 jobs: fast / slow / e2e
├── Dockerfile                 Multi-stage, non-root
├── docker-compose.yml         4 servicios (TimescaleDB, mlflow, migrate, axion-api)
├── Makefile                   ~30 targets
├── requirements.txt
├── .env.example
├── CLAUDE.md                  Contrato con futuras sesiones de Claude Code
├── DEMO.md                    Guía one-page para evaluadores
├── PROJECT_OVERVIEW.md        Este archivo
└── README.md
```

---

## 5. Endpoints HTTP principales

Agrupados por área. Todos respetan RBAC cuando hay JWT/API-key configurado;
todos están sujetos al rate limiter cuando está activo.

**Salud y observabilidad**
- `GET /api/health` — siempre público, exento de RL
- `GET /api/metrics` — Prometheus text format, exento

**Autenticación (Bloque V)**
- `POST /api/auth/login` (público)
- `POST /api/auth/refresh` (público)
- `GET /api/auth/me` (cualquier rol autenticado)

**Escenarios + replay**
- `GET /api/scenarios`
- `POST /api/scenarios/select`              (manager)
- `POST /api/scenarios/upload` (multipart)  (manager)
- `GET /api/state` / `GET /api/process/recent`
- `GET /api/replay/status`
- `POST /api/replay/control`                (operator)

**Recomendaciones + decisiones + outcomes**
- `GET /api/recommendations`
- `GET /api/recommendations/{id}`
- `POST /api/recommendations/{id}/decide`   (operator)
- `GET /api/decisions`
- `GET /api/outcomes/operator`              (Bloque S)
- `GET /api/performance`

**Soft sensor + LSTM + optimizador**
- `GET /api/soft_sensor/purity`
- `GET /api/predictive/forecast`
- `GET /api/optimization/pareto`
- `POST /api/optimization/predict`          (operator)

**Histórico (TimescaleDB)**
- `GET /api/history/scenarios`
- `GET /api/history/samples`
- `GET /api/history/recommendations`
- `GET /api/history/decisions`

**Modelos + drift + reportes**
- `GET /api/models/status`
- `GET /api/drift/status?window=N`
- `GET /api/report/current` (HTML)

**Profile + data source + integración**
- `GET /api/profile`, `POST /api/profile/select`
- `GET /api/data-source/status`, `POST /api/data-source/select`
- `GET /api/integration/opcua/status`
- `GET /api/webhook/status`

**Data versioning (Bloque Z)**
- `GET /api/data/snapshots`
- `GET /api/data/snapshots/{id}`
- `GET /api/data/snapshots/{id}/verify`

**WebSocket**
- `WS /ws/stream` — replay events broadcast en tiempo real

---

## 6. Tests

**787 tests totales** distribuidos así:

- **Unit (`tests/unit/`)**: 735 passing, 13 skipped (slow, opt-in con `--run-slow`)
- **E2E (`tests/e2e/`)**: 16 passing httpx contra uvicorn subprocess; 3 skipped Playwright (opt-in con `--run-browser`)
- **Validation (`tests/validation/`)**: 36 passing — checks de primera ley contra CSVs reales

CI ejecuta:
- `fast` en cada PR y push (Python 3.11 + 3.12)
- `slow` solo en push a main
- `e2e` con TimescaleDB service container, en cada PR y push

---

## 7. Variables de entorno

Todas configurables vía `.env`. Las que vienen seteadas en `.env.example`:

| Variable                       | Default          | Bloque |
| ------------------------------ | ---------------- | ------ |
| `AXION_DB_URL`                 | (vacío)          | C, D   |
| `AXION_API_KEY`                | (vacío, legacy)  | G      |
| `AXION_API_KEY_VIEWER`         | (vacío)          | N      |
| `AXION_API_KEY_OPERATOR`       | (vacío)          | N      |
| `AXION_API_KEY_MANAGER`        | (vacío)          | N      |
| `AXION_JWT_SECRET`             | (vacío)          | V      |
| `AXION_JWT_ACCESS_MINUTES`     | 30               | V      |
| `AXION_JWT_REFRESH_DAYS`       | 7                | V      |
| `AXION_RATE_LIMIT_PER_MIN`     | 120 (0=off)      | W      |
| `AXION_RATE_LIMIT_BURST`       | = per_min        | W      |
| `AXION_PROCESS_PROFILE`        | pilot            | U      |
| `AXION_OPCUA_ENABLED`          | false            | T      |
| `AXION_OPCUA_TAG_MAP`          | (vacío)          | T      |
| `AXION_OPCUA_ENDPOINT`         | (vacío, override)| T      |
| `AXION_OPCUA_USERNAME`         | (vacío)          | T      |
| `AXION_OPCUA_PASSWORD`         | (vacío)          | T      |
| `AXION_OPCUA_SECURITY`         | None             | T      |
| `AXION_OPCUA_CERT_PATH`        | (vacío)          | T      |
| `AXION_OPCUA_KEY_PATH`         | (vacío)          | T      |
| `AXION_OPCUA_TIME_NODE`        | (vacío)          | T      |
| `AXION_WEBHOOK_URL`            | (vacío, off)     | O      |
| `AXION_WEBHOOK_URGENCY`        | critical         | O      |
| `AXION_WEBHOOK_TIMEOUT`        | 5.0              | O      |
| `AXION_WEBHOOK_FORMAT`         | axion            | O      |
| `MLFLOW_TRACKING_URI`          | (vacío)          | I      |

---

## 8. Comandos útiles

```bash
# Testing
make test                    # ~10s — unit fast (735 tests)
make test-slow               # incluye LSTM + NSGA-II (~30s)
make test-e2e                # E2E con uvicorn subprocess (~25s)
make test-e2e-browser        # Playwright (requiere instalación local)
make coverage                # HTML coverage → results/coverage_html/
make smoke                   # spin up + curl + tear down

# Demo single-process (uvicorn local)
make demo                    # arranca DB + migra + serve
make demo-reset              # baja DB y borra volumen

# Stack containerizado completo
make stack-up                # build + 4 servicios docker compose
make stack-logs              # tail logs
make stack-down              # detiene preservando volúmenes
make stack-reset             # detiene y borra volúmenes

# Modelos
make train-soft-sensor       # GBR ensemble → results/models/
make train-lstm              # LSTM forecaster (requiere TF)
make retrain                 # reentrenar + promover si MAE mejoró
make retrain-lstm            # idem para el LSTM (Bloque Q)
make retrain-all             # ambos en secuencia
make mlflow-ui               # UI en :5000

# Validación
make validate-simulator      # checks vs primera ley

# Multi-proceso
make generate-batch-scenarios  # produce los 3 CSVs del batch reactor

# Data versioning
make data-snapshot MSG="..."   # snapshot del data/
make data-snapshots            # listar snapshots
make data-verify ID=<id>       # verificar que data/ matchea

# User management (Bloque V — requiere DB)
make users CMD="create --email a@b.com --role manager"
make users CMD="list"
```

---

## 9. Decisiones que quedaron pendientes

Cosas conscientes que NO están hechas y por qué:

| Item                                | Por qué no está                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| OPC-UA inyectando samples al sim    | Hecho en T'; pero un cliente que conecte una planta real y NO use el simulador no tiene rules    |
| Rules para batch reactor en producción | B01-B03 son starter pack; un deploy real necesita 8-12 reglas afinadas con dominio del proceso |
| Drift retraining loop automatizado  | Hoy: drift se reporta, no dispara retrain. Action would belong in a future "auto-MLOps" block    |
| RBAC fine-grained (resource-level)  | Hoy: 3 roles globales. Cliente regulado podría querer "operator solo puede decidir reglas R1-R3" |
| Multi-tenancy                       | Una sola tabla `process_samples` sin column `tenant_id`. Pilot escala a un solo cliente por instancia |
| Audit log                           | RBAC dice qué podés hacer; un audit log diría qué hiciste. Útil para empresas reguladas (farma)  |
| LSTM retraining via UI              | Hoy es solo CLI. El plant manager no toca CLI                                                    |
| Metrics → Grafana dashboard ejemplo | El endpoint `/api/metrics` está; falta un dashboard JSON que se importe directo en Grafana       |
| Pipeline T2 batch profile drift     | El DriftDetector lee features del active profile; pero la baseline siempre se entrena en `normal.csv` o equivalente |

Ninguna de estas es urgente para una demo. Son material para el siguiente
roadmap si un ICP concreto las pide.

---

## 10. Lo que se puede agregar — propuesta priorizada

Si volvés a tener bandwidth técnico tras una ronda de validación, los
candidatos naturales en orden de impacto:

### Alta prioridad — necesarios para producción real

1. **Audit log** (~M)
   - Tabla `audit_events` con who/what/when/payload_hash
   - Middleware que captura POSTs autenticados
   - `GET /api/audit` solo manager
   - Cierra el loop con RBAC para clientes regulados (farma, alimentos)

2. **Grafana dashboard JSON** (~S)
   - Importable directo en `http://grafana/dashboards`
   - Paneles para latencia p50/p95/p99, request rate, recommendations/min,
     decisions acceptance rate, drift heatmap
   - El work está hecho — `/api/metrics` ya expone todo

3. **LSTM retraining via UI** (~M)
   - Endpoint `POST /api/models/lstm/retrain` (manager)
   - Botón en el panel "HISTORY & MODELS"
   - Status live via WebSocket
   - El backend ya existe (`scripts/retrain.py --model lstm`); solo hay
     que envolverlo en el server y la UI

### Valor técnico

4. **Drift → retrain trigger** (~M)
   - Cuando el drift sostenido cruza el umbral significativo, dispara un
     retrain en background y notifica al manager
   - Cierra el ciclo MLOps completo

5. **Multi-tenancy** (~L)
   - Column `tenant_id` en todas las tablas
   - JWT claim `tenant`
   - Middleware filtra todos los queries
   - Recién vale la pena cuando hayas vendido a 2+ clientes

6. **OPC-UA write-back** (~M)
   - Hoy el `OPCUAWriter` existe pero no está conectado a las decisiones
     UI. Cuando el operador acepta una rec en modo SEMI o AUTONOMOUS,
     debería ejecutar la acción contra el PLC real
   - Requiere safety gates extra

### Polish para evaluadores externos

7. **API docs auto-generadas** (~S)
   - FastAPI ya genera OpenAPI; falta polish: ejemplos en cada endpoint,
     descripciones, agrupación por tags
   - Acceso vía `/docs` y `/redoc`

8. **Demo data seeded en stack-up** (~S)
   - Después de `make stack-up`, automáticamente:
     - Ingestar 3 escenarios al DB
     - Crear un usuario de demo (manager)
     - Tomar un snapshot inicial
   - Reduce friction para un evaluador que abre el repo por primera vez

9. **Healthcheck profundo** (~S)
   - `/api/health` hoy es liveness puro
   - Agregar `/api/health/ready` que valide DB conectada, modelos
     cargados, profile activo, etc.

---

## 11. Glosario rápido

- **Tag**: nombre canónico de una variable de proceso (ej: `cstr.T_R_C`)
- **Profile**: bundle declarativo de tags, KPIs, escenarios, reglas para un proceso (ej: `pilot`, `batch_reactor`)
- **Scenario**: un CSV de datos del proceso bajo una condición específica (ej: `thermal_drift`, `batch_runaway`)
- **Session (analytics)**: secuencia agrupada de alerts del mismo detector + tag
- **Recommendation**: salida del rule engine — diagnóstico + causa + acción + impacto esperado
- **Decision**: lo que el operador hizo con la recomendación (accept/reject/modify)
- **Outcome**: medición real X minutos después de la decisión, comparada con la predicción
- **Snapshot (data)**: manifest content-addressed de los CSVs en `data/` en un momento dado
- **Run (MLflow)**: una corrida de entrenamiento con sus params, métricas y artifacts
- **Live mode (data source)**: cuando el dashboard muestra el stream OPC-UA en lugar del replay
