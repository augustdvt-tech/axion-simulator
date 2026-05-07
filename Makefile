.PHONY: test test-slow test-all test-e2e test-e2e-browser coverage smoke serve clean \
        demo demo-reset demo-db-up demo-migrate \
        train-soft-sensor train-lstm mlflow-ui \
        retrain retrain-force retrain-lstm retrain-all \
        validate-simulator generate-batch-scenarios \
        stack-up stack-down stack-reset stack-logs stack-build \
        data-snapshot data-snapshots data-verify

# ── Default DB URL (matches docker-compose defaults and .env.example) ─────────
AXION_DB_URL ?= postgresql://axion:axion@localhost:5432/axion


# =============================================================================
# Demo targets
# =============================================================================

# One-command demo: start DB, migrate, serve.
# Override DB: AXION_DB_URL=postgresql://... make demo
demo: demo-db-up demo-migrate
	@echo ""
	@echo "┌─────────────────────────────────────────────────────────┐"
	@echo "│  Axion AI — starting server on http://localhost:8000    │"
	@echo "│  Open the dashboard and try /api/health to verify.      │"
	@echo "│  Press Ctrl+C to stop.                                  │"
	@echo "└─────────────────────────────────────────────────────────┘"
	@echo ""
	AXION_DB_URL=$(AXION_DB_URL) uvicorn api.server:app --port 8000

# Start TimescaleDB and wait until it accepts connections (up to 60s).
demo-db-up:
	@command -v docker > /dev/null 2>&1 || { \
	  echo "ERROR: Docker not found. Install Docker Desktop and retry."; exit 1; }
	@echo "Starting TimescaleDB..."
	docker compose up -d
	@echo "Waiting for DB to be ready (up to 60s)..."
	@i=0; until docker compose exec -T timescaledb \
	    pg_isready -U axion -d axion > /dev/null 2>&1; do \
	  i=$$((i+1)); [ $$i -ge 30 ] && echo "ERROR: DB did not start in 60s" && exit 1; \
	  echo "  waiting..."; sleep 2; \
	done
	@echo "DB ready."

# Apply Alembic migrations (idempotent — safe to run multiple times).
demo-migrate:
	@command -v python3 > /dev/null 2>&1 || { echo "ERROR: python3 not found."; exit 1; }
	@echo "Applying DB migrations..."
	AXION_DB_URL=$(AXION_DB_URL) python3 -m alembic -c db/alembic.ini upgrade head
	@echo "Migrations applied."

# Tear down DB container + data volume. Starts completely fresh on next 'make demo'.
# Stop the server with Ctrl+C before running this.
demo-reset:
	@echo "Removing TimescaleDB container and data volume..."
	docker compose down -v
	@echo "Reset complete. Run 'make demo' to start fresh."


# =============================================================================
# Full containerized stack (Bloque Y)
# =============================================================================

# Build the axion-api image (also pulled in by stack-up).
stack-build:
	docker compose build axion-api

# Bring up the full stack: TimescaleDB + MLflow + Axion API. Detached.
# First run takes a couple of minutes (image build + pip install).
stack-up:
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────┐"
	@echo "│  Axion AI — building + starting full stack                   │"
	@echo "│  Dashboard:  http://localhost:8000                           │"
	@echo "│  MLflow UI:  http://localhost:5000                           │"
	@echo "│  TimescaleDB: localhost:5432                                 │"
	@echo "└──────────────────────────────────────────────────────────────┘"
	@echo ""
	docker compose up -d --build
	@echo ""
	@echo "Waiting for axion-api to become healthy..."
	@i=0; while [ $$i -lt 30 ]; do \
	  status=$$(docker compose ps --format json axion-api 2>/dev/null | grep -o '"Health":"[^"]*"' | head -1 || true); \
	  case "$$status" in \
	    *healthy*) echo "  → axion-api healthy"; break ;; \
	    *) i=$$((i+1)); sleep 2; echo "  ...$$i/30" ;; \
	  esac; \
	done
	@echo ""
	@echo "Stack is up. View logs:  make stack-logs"

# Tail logs from all services.
stack-logs:
	docker compose logs -f --tail=50

# Stop the full stack (preserves volumes).
stack-down:
	docker compose down

# Stop and wipe all volumes (DB + MLflow runs).
stack-reset:
	@echo "Removing all containers AND volumes..."
	docker compose down -v
	@echo "Stack reset. Run 'make stack-up' to start fresh."


# =============================================================================
# ML training targets
# =============================================================================

# Run first-principles validation tests (CSTR + column, uses data/ CSVs).
validate-simulator:
	pytest tests/validation/ -v -m validation

# Generate the 3 batch reactor scenario CSVs (used by BATCH_PROFILE).
generate-batch-scenarios:
	python scripts/generate_batch_scenarios.py


# =============================================================================
# Data versioning (Bloque Z)
# =============================================================================

# Take a content-addressed snapshot of every CSV in data/. Pass MSG="..." to
# attach a message:  make data-snapshot MSG="before adding sensor_failure"
data-snapshot:
	python scripts/version_data.py snapshot $(if $(MSG),--message "$(MSG)",)

# List every existing snapshot.
data-snapshots:
	python scripts/version_data.py list

# Verify the current data/ matches a snapshot.  make data-verify ID=<id>
data-verify:
	python scripts/version_data.py verify $(ID)

# Create / list / update users in the `users` table (Bloque V).
# Examples:
#   make users CMD="create --email a@b.com --role manager"
#   make users CMD="list"
users:
	python scripts/users.py $(CMD)

# Retrain soft sensor and promote if MAE improved (compare vs metrics JSON).
# Pass FORCE=1 to promote unconditionally: make retrain FORCE=1
retrain:
	python scripts/retrain.py --model soft_sensor $(if $(FORCE),--force,)

# Unconditional retrain + promote (alias for FORCE=1).
retrain-force:
	python scripts/retrain.py --model soft_sensor --force

# Retrain LSTM forecaster and promote if MAE overall improved.
# Pass FORCE=1 to promote unconditionally: make retrain-lstm FORCE=1
retrain-lstm:
	python scripts/retrain.py --model lstm $(if $(FORCE),--force,)

# Retrain both soft sensor and LSTM in one shot.
retrain-all:
	python scripts/retrain.py --model all $(if $(FORCE),--force,)

# Train the purity soft sensor (GBR ensemble). Logs to MLflow if available.
train-soft-sensor:
	python examples/train_soft_sensor.py

# Train the LSTM multi-horizon forecaster. Requires TensorFlow. Logs to MLflow.
train-lstm:
	python examples/train_lstm_forecaster.py

# Launch the MLflow tracking UI (default: http://127.0.0.1:5000).
# Reads ./mlruns by default; override with MLFLOW_TRACKING_URI.
mlflow-ui:
	mlflow ui --port 5000


# =============================================================================
# Development targets
# =============================================================================

# Fast unit suite (~15s). Skips LSTM training and NSGA-II runs.
test:
	@if [ ! -d tests ]; then echo "No tests/ directory found — skipping."; else pytest tests/unit/ -q; fi

# Full suite including slow tests (LSTM, NSGA-II, ~30s).
test-slow:
	@if [ ! -d tests ]; then echo "No tests/ directory found — skipping."; else pytest tests/unit/ --run-slow -q; fi

test-all: test-slow

# End-to-end suite: spawns a real uvicorn subprocess and drives it via httpx.
# Slower (~10-20s of startup) but catches integration bugs that TestClient misses.
test-e2e:
	pytest tests/e2e/ -m e2e -v

# Browser suite (Playwright). Requires `pip install pytest-playwright` and
# `playwright install chromium`. Gated behind --run-browser.
test-e2e-browser:
	pytest tests/e2e/ -m browser --run-browser -v

# HTML coverage report written to results/coverage_html/
coverage:
	@if [ ! -d tests ]; then \
		echo "No tests/ directory found — skipping."; \
	else \
		pytest tests/unit/ --run-slow \
			--cov=. \
			--cov-report=term-missing \
			--cov-report=html:results/coverage_html \
			-q && echo "Coverage report: results/coverage_html/index.html"; \
	fi

# Start the server, hit key endpoints, then stop. Exits non-zero on failure.
smoke:
	@bash scripts/smoke_test.sh

# Dev server with auto-reload on port 8000 (no DB required).
serve:
	uvicorn api.server:app --reload --port 8000

# Remove Python cache artifacts and test/coverage outputs.
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage results/coverage_html
	@echo "Clean done."
