# Axion AI — Demo Guide

Sistema autónomo de ingeniería de procesos: monitoreo en tiempo real, detección de anomalías, recomendaciones y forecasting predictivo para una planta piloto (CSTR + columna de destilación).

---

## Prerequisites

| Herramienta | Versión mínima | Verificar |
|-------------|---------------|-----------|
| Python      | 3.9+          | `python3 --version` |
| Docker Desktop | cualquiera | `docker --version` |
| pip packages | — | `pip3 install -r requirements.txt` |

---

## Iniciar la demo (un comando)

```bash
make demo
```

Esto hace automáticamente:
1. Levanta TimescaleDB en Docker
2. Espera que la DB acepte conexiones
3. Aplica las migraciones Alembic
4. Inicia el server en `http://localhost:8000`

El dashboard se abre en **`http://localhost:8000`**.

> **Sin Docker?** `make serve` inicia el server en modo memoria (sin persistencia).

---

## Qué ver en la demo

### Dashboard — `http://localhost:8000`

El servidor carga el escenario `thermal_drift` y reproduce los datos a 60× (1 hora simulada = 1 minuto real). El dashboard muestra:

- **Gráficos en tiempo real**: temperatura de reactor, pureza de producto, reboiler duty
- **Recomendaciones**: aparecen a medida que el sistema detecta anomalías
- **Decisiones del operador**: podés aprobar, modificar o rechazar cada recomendación
- **Soft sensor**: predicción de pureza por GBR ensemble vs. valor medido
- **Pareto front**: `http://localhost:8000/api/optimization/pareto`

### Cambiar escenario

```bash
curl -X POST http://localhost:8000/api/scenarios/select \
     -H "Content-Type: application/json" \
     -d '{"scenario": "feed_perturbation"}'
```

Escenarios disponibles: `normal`, `thermal_drift`, `feed_perturbation`, `high_purity_demand`, `startup_ramp`, `cooling_failure`, `composition_shift`.

### Endpoints REST clave

| Endpoint | Descripción |
|----------|-------------|
| `GET /api/health` | Liveness check |
| `GET /api/scenarios` | Lista de escenarios |
| `GET /api/state` | Snapshot actual del proceso |
| `GET /api/recommendations` | Recomendaciones emitidas hasta ahora |
| `GET /api/decisions` | Log de decisiones |
| `GET /api/replay/status` | Reloj de replay y progreso |
| `WS  /ws/stream` | Stream WebSocket en tiempo real |

---

## Autenticación (opcional)

Por defecto el server corre sin auth. Para habilitarla:

```bash
# Iniciar con API key
AXION_API_KEY=mi-clave-secreta make demo

# Llamar endpoints autenticados
curl -H "X-API-Key: mi-clave-secreta" http://localhost:8000/api/state

# WebSocket (query param)
ws://localhost:8000/ws/stream?api_key=mi-clave-secreta
```

`/api/health` siempre es público (sin auth) para facilitar liveness checks.

---

## Resetear a estado inicial

```bash
# 1. Detener el server con Ctrl+C
# 2. Luego:
make demo-reset

# 3. Volver a empezar:
make demo
```

`demo-reset` baja el container de TimescaleDB y elimina el volumen de datos. La próxima vez que corras `make demo` empieza desde cero.

---

## Stack completo containerizado (`make stack-up`)

Para evaluadores que quieren un único `docker compose up` con todo —
TimescaleDB + MLflow + el server Axion empaquetado — sin instalar Python
ni dependencias localmente:

```bash
make stack-up         # build + up (TimescaleDB, MLflow, Axion API)
make stack-logs       # tail logs
make stack-down       # detiene todo (preserva los volúmenes)
make stack-reset      # detiene todo y borra los volúmenes
```

Servicios expuestos:
- `http://localhost:8000` — Dashboard + API Axion
- `http://localhost:5000` — MLflow tracking UI
- `localhost:5432` — TimescaleDB

La imagen `axion-api` se construye desde `Dockerfile` (multi-stage, Python
3.11-slim, non-root). El service `migrate` corre `alembic upgrade head`
una sola vez y bloquea el arranque de `axion-api` hasta que termina.

---

## Arquitectura rápida

```
simulator/     → CSTR + columna de destilación (integración RK4)
analytics/     → SPC, PCA, Trend, FrozenSensor (detección de anomalías)
recommendations/ → 10 reglas R1–R10 + engine (diagnóstico + acción)
consensus/     → Modos ADVISOR/SEMI/AUTONOMOUS + safety gate
soft_sensor/   → GBR ensemble para predicción de pureza
predictive/    → LSTM multi-horizonte (5/15/30/60 min)
optimizer/     → NSGA-II + ProcessSurrogate + 4 objetivos
api/server.py  → FastAPI + WebSocket (replay en tiempo real)
db/            → TimescaleDB (Alembic migrations + DbClient)
```
