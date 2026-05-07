# Axion AI — MVP completo con Operator Console

Plataforma de desarrollo del MVP funcional completo de Axion AI. Cubre cinco tareas del roadmap:

1. **Simulador de proceso** (`simulator/`) — CSTR + columna de destilación, 8 escenarios
2. **Motor analítico** (`analytics/`) — 5 detectores + EventSessions
3. **Sistema de recomendaciones** (`recommendations/`) — 8 reglas diagnósticas
4. **Capa de consenso** (`consensus/`) — decisiones, outcomes, 3 modos operativos
5. **Operator Console** (`api/` + `ui/`) — REST + WebSocket + UI React industrial

Diseñado para ser **modular**, **escalable** y **reproducible**.

---

## Instalación

```bash
cd axion_simulator
pip install -r requirements.txt
```

Dependencias: `numpy`, `pandas`, `scipy`, `matplotlib`, `fastapi`, `uvicorn`, `websockets`.

Las dependencias JavaScript del frontend están incluidas localmente en `ui/vendor/` — no requieren internet para funcionar.

---

## Uso

### Opción 1 — Operator Console (recomendado)

Arranca el server con UI integrada:

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Luego abre en el navegador: **http://localhost:8000**

El server automáticamente:
- Entrena el motor analítico con `data/normal.csv`
- Carga el escenario activo (default: `thermal_drift`)
- Pre-computa todas las sesiones, recomendaciones, decisiones y outcomes
- Empieza a replay a 60× (configurable en la UI)
- Expone la UI React con dark mode industrial

### Opción 2 — Pipeline batch (sin UI)

Si querés correr el pipeline completo como batch y generar reportes:

```bash
python run_simulation.py --scenario all              # generar datos
python examples/benchmark_consensus.py               # pipeline completo
```

Genera tablas y figuras en `results/`.

---

## Operator Console

El console implementa los tres principios del documento de arquitectura:

1. **Mínima carga cognitiva** — el ingeniero entiende el estado del proceso en menos de 3 segundos
2. **Información por excepción** — la pantalla está tranquila cuando todo está bien
3. **Acción a un click** — Accept / Modify / Reject desde cualquier recomendación

### Layout

- **Top bar**: brand, scenario picker, sim clock con progreso, controles de replay (play / pause / restart / speed 60×-1800×), indicador LIVE
- **Panel izquierdo — Live Process State**: tiles industriales con valores del CSTR (T_R, T_J, C_A, F_cool) y de la columna (Purity B, RR, T_bot, Q_reb). Cada tile se colorea según rangos operacionales (verde OK, ámbar warn, rojo alarm). Chart de tendencia de últimas 4 horas con T_R y pureza en ejes duales.
- **Panel central — Recommendation Queue**: lista cronológica de recomendaciones, agrupadas por urgencia. Cada card muestra urgency badge, diagnóstico, regla disparada, confidence, priority score, y status (pending / accepted / modified / rejected).
- **Panel derecho — Recommendation Detail**: al seleccionar una card, despliega diagnosis, probable cause, recommended action con comparación visual current → proposed, expected impact por variable con ETA, y el estado de la decisión. Si está pending, muestra botones grandes Accept / Modify / Reject.

### Paleta y tipografía

Pensados para turnos largos frente a la pantalla:
- **Dark mode** casi-negro (`#07090d`) con scanlines sutiles
- **Cyan** (`#4dd2ff`) como acento primario / nominal
- **Amber / signal red** para atención y alarmas
- **JetBrains Mono** para datos técnicos, **Space Grotesk** para display

### Endpoints REST (para integraciones externas)

```
GET   /api/health                       - liveness
GET   /api/scenarios                    - lista de escenarios
POST  /api/scenarios/select             - cambiar escenario activo
GET   /api/state                        - snapshot actual del proceso
GET   /api/process/recent?samples=240   - últimos N samples
GET   /api/recommendations?limit=80     - recomendaciones visibles
GET   /api/recommendations/{id}         - detalle
POST  /api/recommendations/{id}/decide  - registrar decisión del operador
GET   /api/decisions                    - log de decisiones
GET   /api/performance                  - track record por regla
GET   /api/replay/status                - estado del replay clock
POST  /api/replay/control               - play / pause / speed / seek / restart
```

### WebSocket

`ws://localhost:8000/ws/stream` empuja en tiempo real:
- `snapshot` — al conectar: estado actual
- `tick` — cada segundo: sample + nuevos eventos
- `decision_recorded` — al aceptar/rechazar desde UI
- `scenario_changed` — al cambiar escenario

---

## Arquitectura

```
axion_simulator/
├── simulator/        # Proceso industrial simulado
├── analytics/        # Motor analítico de detección
├── recommendations/  # Sistema de recomendaciones (reglas)
├── consensus/        # Capa de consenso humano-máquina
├── api/              # FastAPI server (REST + WebSocket)
│   └── server.py
├── ui/               # Operator Console (React + Tailwind + Recharts)
│   ├── index.html
│   └── vendor/       # Dependencias JS locales (sin internet)
├── examples/         # Benchmarks end-to-end
├── data/             # CSVs del simulador
└── results/          # Outputs analíticos
```

---

## Pipeline conceptual

```
Sensor data  →  AnalyticalEngine  →  RecommendationEngine  →  ConsensusController
   (CSV)          (5 detectores)         (8 reglas)          (3 modos + safety)
                        ↓                      ↓                      ↓
                 EventSessions          Recommendations       DecisionLog + Outcomes
                                                                      ↓
                                                               PerformanceTracker
                                                             (confidence learning)

Todo lo anterior es consumido en tiempo real por:
   FastAPI server → WebSocket → React Operator Console
```

---

## Escenarios disponibles

| Escenario | Duración | Qué muestra |
|---|---|---|
| `normal` | 24 h | Línea base estable |
| `thermal_drift` | 72 h | Ensuciamiento gradual de camisa (R01) |
| `feed_perturbation` | 24 h | Step en composición (R02, R06) |
| `reactor_instability` | 24 h | PID mal sintonizado (R03) |
| `quality_degradation` | 96 h | Pérdida de volatilidad (R04, R06) |
| `energy_waste` | 24 h | Reflujo excesivo (R05, optimización) |
| `product_grade_change` | 24 h | Cambio de especificación (R08) |
| `sensor_failure` | 24 h | Sensor congelado (R07) |

Para ver la UI en acción con el escenario más visual, probar `feed_perturbation` a 1800× — en ~15 segundos ves aparecer recomendaciones MEDIUM, HIGH y CRITICAL en secuencia.

---

## Roadmap

- [x] Tarea 1: Documento de arquitectura ✅
- [x] Tarea 2: Simulador modular con 8 escenarios ✅
- [x] Tarea 3: Motor analítico (SPC, PCA, Trend, CUSUM, FrozenSensor) ✅
- [x] Tarea 4: Sistema de recomendaciones (10 reglas) ✅
- [x] Tarea 5: Capa de consenso humano-máquina ✅
- [x] Tarea 6: Soft sensor ML para pureza del producto ✅
- [x] Tarea 7: LSTM Forecaster multi-horizonte ✅
- [x] Tarea 8: Optimizador multiobjetivo (NSGA-II + Pareto front) ✅
- [x] Tarea 9: Operator Console (REST + WebSocket + React UI) ✅
- [x] Tarea 10: Conector OPC-UA para datos reales de PLC/DCS ✅

**🎯 Roadmap completo — las 10 tareas del documento de arquitectura están cerradas.**

---

## LSTM Forecaster (Tarea 7)

El módulo `predictive/` proporciona forecasting multi-horizonte y multi-target de las variables críticas del proceso. Anticipa excursiones operativas antes de que ocurran, complementando el `TrendDetector` lineal con dinámicas no-lineales aprendidas.

### Arquitectura

| Módulo | Responsabilidad |
|---|---|
| `windowing.py` | Sliding-window builders + time-aware split + Z-normalization scaler |
| `lstm.py` | `LSTMForecaster` two-layer LSTM (48→24) + Dense head, multi-target multi-horizon |
| `detector.py` | `LSTMPredictiveDetector` que emite alerts con `time-to-violation` |

### Modelo

- 2 capas LSTM (48 y 24 hidden units) con dropout
- Multi-output: predice 4 targets simultáneamente
- 4 horizontes: **5, 15, 30, 60 minutos**
- Lookback: 60 minutos
- Loss MSE en espacio normalizado, optimizer Adam con ReduceLROnPlateau
- EarlyStopping con `patience=5`, `restore_best_weights=True`

### Features (12)

- Endogenous: `cstr.T_R_C`, `column.purity_B`, `column.Q_reb_kW`, `cstr.conversion`
- Manipulated: `column.RR`, `cstr.F_cool`, `cstr.F_feed`
- Disturbances: `cstr.C_A`, `cstr.T_feed_C`
- Auxiliary: `column.T_bot_C`, `column.T_top_C`, `cstr.T_J_C`

### Targets (4)

- `cstr.T_R_C` — temperatura del reactor (control térmico)
- `column.purity_B` — calidad del producto
- `column.Q_reb_kW` — energía
- `cstr.conversion` — performance reactor

### Performance

Entrenando con 7 escenarios (13,158 train / 3,289 val):

| Target | t+5min | t+15min | t+30min | t+60min |
|---|---|---|---|---|
| **T_R_C** (R²) | 0.964 | 0.964 | 0.968 | 0.970 |
| **purity_B** (R²) | 0.936 | 0.924 | 0.941 | 0.920 |
| **Q_reb_kW** (R²) | 0.910 | 0.892 | 0.903 | 0.858 |
| **conversion** (R²) | 0.471 | 0.477 | 0.477 | 0.480 |

R² consistentemente alto y *estable* a través de horizontes — el modelo aprendió dinámicas reales (no copia el valor actual).

### Regla R10_PredictedExcursion

Convierte forecasts en recomendaciones operativas con time-to-violation. Mapping:
- `purity_B` ↓ spec → ↑ Reflux Ratio
- `T_R_C` ↑ limit → ↑ Cooling Flow
- `Q_reb_kW` ↑ limit → ↓ Reflux Ratio (within purity constraints)

Urgency desde `horizon_minutes`:
- ≤10 min: HIGH
- ≤30 min: MEDIUM  
- > 30 min: LOW

### Endpoints API

```
GET /api/predictive/forecast   - current values, per-horizon points (5/15/30/60),
                                  full trajectory de 60 puntos, metadata
```

### Visualización

Nueva sección **"AI FORECAST · LSTM"** en el panel del proceso con tiles:
- **PURITY B**: valor actual + 4 mini-tiles (5/15/30/60 min) con color rojo si violan spec
- **T_REACTOR**: idem con threshold 82°C
- Indicador `⚠ +Nm` en el header del tile cuando se predice violación

### Entrenamiento

```bash
python examples/train_lstm_forecaster.py
# → results/models/lstm_forecaster/  (model.keras + meta.joblib)
# → results/lstm_training_history.png
# → results/lstm_horizon_metrics.png
```

### Benchmark visual

```bash
python examples/benchmark_lstm.py
# → results/lstm_forecasts.png
```

Genera figura con sliding-window predictions vs actual en 4 escenarios × 4 targets.

### Uso programático

```python
from predictive import LSTMForecaster
from pathlib import Path

forecaster = LSTMForecaster.load(Path("results/models/lstm_forecaster"))
# Predict from a DataFrame (uses last lookback_steps as input)
predictions = forecaster.predict_at_horizons(df_recent, horizons_minutes=[5, 15, 30])
# Returns: { "column.purity_B": {5: 98.4, 15: 98.2, 30: 98.0}, ... }
```

---

## Optimizador Multiobjetivo (Tarea 8)

El módulo `optimizer/` propone puntos de operación óptimos balanceando objetivos contradictorios típicos de procesos industriales: **calidad vs energía vs producción vs estabilidad**. Esto cierra el ciclo de Axion AI desde detección/diagnóstico hacia prescripción autónoma.

### Arquitectura

| Módulo | Responsabilidad |
|---|---|
| `surrogate.py` | `ProcessSurrogate` analítico (Arrhenius CSTR + Fenske-Underwood-Gilliland) |
| `objectives.py` | `Objective` ABC + 4 implementaciones concretas (purity, energy, production, stability) |
| `nsga2.py` | `NSGA2Optimizer` algoritmo genético multiobjetivo, ~250 líneas, sin dependencias externas |

### Por qué surrogate analítico (y no ML)

Los CSVs históricos del simulador contienen *perturbaciones de estado* (drift, instability, sensor failures), no *exploración de setpoints*: las MVs (RR, F_cool, F_feed) varían muy poco. Un modelo ML entrenado sobre ellos da R²≈0.34 — no aprende las relaciones causa-efecto que el optimizador necesita.

En cambio, el modelo analítico reducido captura la física correctamente: Arrhenius para la cinética del CSTR, Fenske-Underwood-Gilliland para la separación de la columna binaria. Ventajas:
- **Generaliza fuera del envelope histórico** — el optimizador puede explorar setpoints que nunca se probaron en planta
- **Determinístico y explicable** — un ingeniero verifica las correlaciones contra fórmulas de libro de texto
- **No requiere entrenamiento** — solo calibración de constantes (k₀, Eₐ, UA, α)
- **Fast** — evalúa miles de candidatos en segundos

Validado: sweep de RR (3.5 → 7.5) produce las sensitividades correctas: pureza sube monotónicamente (97.11% → 98.55%), Q_reb sube monotónicamente (154 → 290 kW).

### NSGA-II custom (sin dependencias)

Implementación a mano del Non-dominated Sorting Genetic Algorithm II:
- Fast non-dominated sort (rank 0 = frente de Pareto)
- Crowding distance para preservar diversidad
- Tournament selection binario por (rank, -crowding)
- SBX-style blend crossover + gaussian mutation escalada al rango
- Determinístico con seed

Configuración default: 60 generaciones × 80 individuos → 80 puntos en el frente de Pareto, < 5 segundos.

### Endpoints API

```
GET  /api/optimization/pareto                  → frente de Pareto + nominal + current
GET  /api/optimization/pareto?refresh=true     → fuerza recompute
POST /api/optimization/predict {RR, F_cool...} → KPIs predichos para un setpoint manual
```

### Visualización en el Panel de Optimización

Cuando no hay recomendación seleccionada, el panel derecho de la console muestra:
- **Purity vs Energy chart**: curva del Pareto front con línea de spec
- **Current vs Pareto-Suggested table**: comparación lado a lado del operating point actual con el "best feasible" del frente
- **Delta indicator**: cuánto cambia purity y energy si el operador adopta la sugerencia
- **Setpoint trajectory chart**: cómo varían RR/F_cool/F_feed a lo largo del frente

### Uso programático

```python
from optimizer import (
    ProcessSurrogate, NSGA2Optimizer,
    PurityObjective, EnergyObjective, ProductionObjective, StabilityObjective,
)

surrogate = ProcessSurrogate()    # analytical, no training needed
bounds = {
    "column.RR":   (3.0, 7.5),
    "cstr.F_cool": (0.10, 0.55),
    "cstr.F_feed": (1.7, 2.3),
}
fixed = {"cstr.C_A": 157.0, "cstr.T_feed_C": 70.0}

optimizer = NSGA2Optimizer(
    surrogate=surrogate,
    objectives=[PurityObjective(spec=98.5), EnergyObjective(),
                ProductionObjective(), StabilityObjective()],
    bounds=bounds, fixed_inputs=fixed, seed=42,
)
front = optimizer.run(n_generations=60, population_size=80)
# front is a list of OperatingPoint with .inputs, .kpis, .objectives
```

### Benchmark visualization

```bash
python examples/optimize_pilot.py
# → results/optimization_pareto.png
# → results/optimization_setpoints.csv
# → results/models/process_surrogate.joblib
```

Genera figura de 2 paneles: frente de Pareto purity-vs-energy con producción colorada, y trayectorias de los 3 setpoints a lo largo del frente.

---

## Soft Sensor ML (Tarea 6)

El módulo `soft_sensor/` proporciona estimación continua de la pureza del producto a partir de variables secundarias del proceso, eliminando la dependencia del cromatógrafo de gases (típicamente 1 muestra cada 15-30 minutos en plantas reales).

### Arquitectura

| Módulo | Responsabilidad |
|---|---|
| `base.py` | `SoftSensor` ABC, `SoftSensorMetrics`, helpers de evaluación |
| `purity.py` | `PuritySoftSensor` con ensemble de Gradient Boosting + uncertainty |
| `detector.py` | `SoftSensorDetector` que detecta divergencia sostenida sensor real vs predicción |

### Features (basadas en física de destilación)

- **`column.T_bot_C`** — temperatura del fondo, predictor dominante (98.5% feature importance)
- `column.T_top_C` — pareja con T_bot, fingerprint del perfil térmico
- `column.RR` — variable manipulada principal
- `column.Q_reb_kW` — energía aportada
- `column.F_vap_kgh` — flujo de vapor
- `column.P_top_bar` — presión, afecta volatilidad relativa
- `cstr.C_A` — perturbación aguas arriba

### Performance

Entrenando con 7 escenarios (17280 samples, split 80/20):

| Métrica | Valor |
|---|---|
| MAE | 0.150 % |
| RMSE | 0.225 % |
| R² | 0.997 |
| Bias | +0.001 |
| Max error | 2.006 % |

Mejor que la tolerancia típica de un cromatógrafo industrial (±0.3-0.5%).

### Ensemble + uncertainty

El soft sensor entrena 5 modelos GradientBoosting con bootstrap sampling y diferentes seeds. La predicción es la media del ensemble; el intervalo ±2σ refleja la dispersión del ensemble — crece automáticamente cuando el operating point está fuera del envelope de entrenamiento, lo cual le indica al operador cuándo dejar de confiar en la predicción.

### Regla R09_SoftSensorDivergence

Detecta cuando la medición real difiere sostenidamente de la predicción del soft sensor. Dos interpretaciones operativas posibles:
1. **Instrumento mal calibrado** — la medición es errónea, el modelo está bien
2. **Proceso fuera del envelope de entrenamiento** — extrapolación del modelo

La acción recomendada (`VERIFY_INSTRUMENT`) le da al operador el contexto para distinguir entre ambas.

### Uso programático

```python
from soft_sensor import SoftSensor, SoftSensorDetector
from analytics import AnalyticalEngine
from pathlib import Path

# Cargar modelo entrenado
sensor = SoftSensor.load(Path("results/models/purity_soft_sensor.joblib"))

# Predicción individual
prediction, uncertainty = sensor.predict_with_confidence(df_features)

# Integrar al motor analítico
ss_detector = SoftSensorDetector(
    sensor=sensor, target_tag="column.purity_B",
    abs_tolerance=0.5, min_duration_minutes=10.0,
)
ae = AnalyticalEngine(extra_detectors=[ss_detector])
ae.fit(df_train)
sessions = ae.run_sessions(df_eval)   # incluye divergencias soft-sensor
```

### Endpoints API

```
GET /api/soft_sensor/purity?samples=240   - serie temporal predicción + ±2σ + medición
GET /api/state                            - incluye campo soft_sensor con valor live
```

### Visualización en la Console

Nueva sección "AI SOFT SENSOR" en el panel izquierdo con:
- Tile **PURITY B · PREDICTED** mostrando valor, intervalo ±2σ, residual
- **AGREE %** con punto pulsante (verde >95%, amber 80-95%, rojo <80%)
- Overlay magenta en el chart de tendencia con banda de confianza

### Entrenamiento

```bash
python examples/train_soft_sensor.py
# → results/models/purity_soft_sensor.joblib
# → results/soft_sensor_predictions.png
```

### Benchmark de valor operativo

```bash
python examples/benchmark_soft_sensor_value.py
# → results/soft_sensor_value.png
```

Compara detection latency con/sin soft sensor en escenarios de:
- Sensor sano (control)
- Sensor con drift de 0.15%/h (miscalibración realista que no es detectable como "frozen")

Resultado: el soft sensor permite detectar drift de calibración del cromatógrafo **antes** de que QC lo detecte aguas abajo.

---

## Integración OPC-UA (Tarea 10)

El módulo `integration/` conecta Axion AI a cualquier servidor OPC-UA estándar (PLCs Siemens / Allen-Bradley / Schneider, DCS Honeywell / ABB / Emerson, o historiadores con wrapper OPC-UA). Es la puerta de entrada a una planta real.

### Componentes

| Módulo | Responsabilidad |
|---|---|
| `tag_map.py` | Mapeo declarativo node_id OPC-UA ↔ tag canónico Axion (único archivo plant-specific) |
| `opcua_source.py` | Cliente que polea tags, valida rango/staleness, auto-reconecta |
| `opcua_writer.py` | Escritor de setpoints con 3 salvaguardas (whitelist / range / readback) |
| `ingestion.py` | Puente stream-to-batch: buffer rolling + evaluación periódica del pipeline |
| `opcua_mock_server.py` | Servidor OPC-UA de prueba que replica los CSVs como nodos reales |

### Modelo de datos OPC-UA publicado por el mock server

```
/Objects
    /Axion
        /CSTR01
            T_R, T_J, C_A, X, P_R, T_feed, T_cool_in      (read-only)
            F_feed_SP, F_cool_SP                          (writable)
        /COL01
            PurityB, x_D, x_B_A, T_top, T_bot, Q_reb,
            F_vap, P_top, P_bot                           (read-only)
            RR_SP                                         (writable)
        SIM_TIME                                          (simulated epoch time)
```

### Demo end-to-end

```bash
# Arranca el mock server + source + ingestion + writer, todo sobre OPC-UA real
python examples/opcua_e2e_demo.py
```

Validado: **734 samples capturados, 165 recomendaciones emitidas en vivo durante 90s de wall-clock sobre 13h de tiempo de proceso simulado**. El writer valida correctamente:
- `SUCCESS` al escribir column.RR autorizado con readback match
- `BLOCKED_NOT_WRITABLE` al intentar escribir un tag read-only
- `BLOCKED_OUT_OF_RANGE` al exceder los límites del tag map

### Figura de validación

```bash
python examples/opcua_e2e_figure.py
```

Genera `results/opcua_e2e_figure.png` mostrando captura live de samples y timeline de recomendaciones emitidas.

### Deployment en planta real

Para apuntar Axion AI a una planta real, el único archivo a editar es `integration/tag_map.py`:
- Cambiar `server.endpoint` al servidor OPC-UA de planta
- Reemplazar los `node_id` con los identificadores reales del DCS/SCADA
- Ajustar `writable` y `min_range/max_range` según la política del plant engineer

Nada más en el sistema es plant-specific — analytics, reglas, consensus y UI permanecen idénticos entre sites.

---

## Licencia

Uso interno — proyecto Axion AI.
