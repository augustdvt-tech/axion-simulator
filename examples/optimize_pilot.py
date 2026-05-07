"""
Axion AI - Multi-Objective Optimization Benchmark
=================================================

Trains the process surrogate from combined scenario data, runs NSGA-II
to find the Pareto front of (purity vs energy), and visualizes the result.

The Pareto front is the operationally meaningful output: every point on
the front represents a setpoint combination that no other point dominates.
The operator chooses where on the front to operate based on current
business priorities (energy cost, contractual purity, throughput).

Outputs:
    - results/models/process_surrogate.joblib   trained surrogate
    - results/optimization_pareto.png           Pareto front visualization
    - results/optimization_setpoints.csv        full Pareto set with setpoints
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from optimizer import (
    ProcessSurrogate, NSGA2Optimizer,
    PurityObjective, EnergyObjective, ProductionObjective, StabilityObjective,
)

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR  = RESULTS_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_SCENARIOS = [
    "normal", "thermal_drift", "feed_perturbation",
    "reactor_instability", "quality_degradation",
    "energy_waste", "product_grade_change",
]


def load_combined() -> pd.DataFrame:
    frames = []
    for s in TRAIN_SCENARIOS:
        df = pd.read_csv(DATA_DIR / f"{s}.csv")
        df["_scenario"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    print("=" * 72)
    print("Axion AI — Multi-Objective Optimization Benchmark")
    print("=" * 72)

    # ---- 1. Train the surrogate ----
    print("\n1) Training process surrogate...")
    df = load_combined()
    surrogate = ProcessSurrogate()
    metrics = surrogate.fit(df)
    print(f"   Surrogate trained on {len(df)} samples.")
    print(f"   Per-output metrics:")
    print(metrics.format())

    surrogate.save(MODELS_DIR / "process_surrogate.joblib")
    print(f"   Saved: {MODELS_DIR / 'process_surrogate.joblib'}")

    # ---- 2. Define optimization context ----
    print("\n2) Setting up optimization...")
    # Bounds: from safety limits + envelope
    bounds = {
        "column.RR":    (3.0, 7.5),
        "cstr.F_cool":  (0.10, 0.55),
        "cstr.F_feed":  (1.7, 2.3),
    }
    # Disturbance variables fixed at typical normal-operation values
    fixed = {
        "cstr.C_A":      157.0,
        "cstr.T_feed_C": 70.0,
    }

    objectives = [
        PurityObjective(weight=1.0, spec=98.5),
        EnergyObjective(weight=1.0),
        ProductionObjective(weight=0.7),
        StabilityObjective(weight=0.3),
    ]
    print(f"   Objectives: {[o.name for o in objectives]}")
    print(f"   Decision variables: {list(bounds.keys())}")
    print(f"   Fixed inputs: {fixed}")

    # ---- 3. Run NSGA-II ----
    print("\n3) Running NSGA-II...")
    optimizer = NSGA2Optimizer(
        surrogate=surrogate,
        objectives=objectives,
        bounds=bounds,
        fixed_inputs=fixed,
        seed=42,
    )
    pareto = optimizer.run(n_generations=60, population_size=80)
    print(f"   Pareto front: {len(pareto)} non-dominated solutions found")

    # ---- 4. Compare against baseline (current operating point) ----
    print("\n4) Pareto front summary (top 10 by purity):")
    pareto_sorted = sorted(pareto, key=lambda p: -p.objectives["purity"])
    print(f"   {'rank':>4s}  {'RR':>5s}  {'F_cool':>7s}  {'F_feed':>7s}  "
          f"{'purity':>7s}  {'Q_reb':>7s}  {'prod':>6s}  {'T_R':>6s}")
    print("   " + "-" * 75)
    for i, p in enumerate(pareto_sorted[:10]):
        print(f"   {i+1:>4d}  "
              f"{p.inputs['column.RR']:>5.2f}  "
              f"{p.inputs['cstr.F_cool']:>7.3f}  "
              f"{p.inputs['cstr.F_feed']:>7.2f}  "
              f"{p.kpis['column.purity_B']:>7.2f}  "
              f"{p.kpis['column.Q_reb_kW']:>7.1f}  "
              f"{p.objectives['production']:>6.2f}  "
              f"{p.kpis['cstr.T_R_C']:>6.2f}")

    # Reference: nominal operation (RR=5.5, F_cool=0.30, F_feed=2.0)
    nominal = surrogate.predict_one(
        **{"column.RR": 5.5, "cstr.F_cool": 0.30, "cstr.F_feed": 2.0,
           "cstr.C_A": fixed["cstr.C_A"], "cstr.T_feed_C": fixed["cstr.T_feed_C"]}
    )
    print(f"\n   Nominal operating point (RR=5.5, F_cool=0.30, F_feed=2.0):")
    print(f"     purity={nominal['column.purity_B']:.2f}%  Q_reb={nominal['column.Q_reb_kW']:.1f}kW")

    # Save the Pareto set
    rows = []
    for p in pareto:
        row = {**p.inputs, **p.kpis, **{f"obj_{k}": v for k, v in p.objectives.items()}}
        rows.append(row)
    pareto_df = pd.DataFrame(rows)
    pareto_df.to_csv(RESULTS_DIR / "optimization_setpoints.csv", index=False)
    print(f"   Pareto setpoints saved: {RESULTS_DIR / 'optimization_setpoints.csv'}")

    # ---- 5. Visualize ----
    print("\n5) Generating visualization...")
    purity_vals = np.array([p.objectives["purity"] for p in pareto])
    energy_vals = np.array([p.objectives["energy"] for p in pareto])
    prod_vals   = np.array([p.objectives["production"] for p in pareto])
    rr_vals     = np.array([p.inputs["column.RR"] for p in pareto])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: Purity vs Energy with production color ---
    ax = axes[0]
    sc = ax.scatter(purity_vals, energy_vals, c=prod_vals,
                    cmap="viridis", s=60, edgecolors="k", linewidths=0.5)
    ax.scatter([nominal["column.purity_B"]], [nominal["column.Q_reb_kW"]],
               marker="*", s=300, color="red", edgecolors="black",
               linewidths=1.0, label="Nominal operation", zorder=5)
    ax.set_xlabel("Purity B [%]", fontsize=11)
    ax.set_ylabel("Reboiler duty Q_reb [kW]", fontsize=11)
    ax.set_title("Pareto front — Purity vs Energy\n(color = production rate)",
                 fontsize=11, loc="left")
    ax.axvline(98.5, color="red", ls=":", lw=0.8, alpha=0.5, label="spec ≥ 98.5%")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Production rate [kmol/h B]", fontsize=10)

    # --- Right panel: 3 setpoints across the Pareto front ---
    ax = axes[1]
    # Sort the front by purity for a clean curve
    order = np.argsort(purity_vals)
    x = purity_vals[order]
    rr = rr_vals[order]
    fcool_vals = np.array([p.inputs["cstr.F_cool"] for p in pareto])[order]
    ffeed_vals = np.array([p.inputs["cstr.F_feed"] for p in pareto])[order]

    ax.plot(x, rr, "-o", color="#2980b9", lw=1.5, ms=4, label="Reflux RR")
    ax2 = ax.twinx()
    ax2.plot(x, fcool_vals * 10, "-s", color="#27ae60", lw=1.5, ms=4,
             label="F_cool ×10 [m³/h]", alpha=0.7)
    ax2.plot(x, ffeed_vals,    "-^", color="#e67e22", lw=1.5, ms=4,
             label="F_feed [m³/h]", alpha=0.7)
    ax.set_xlabel("Purity B [%]", fontsize=11)
    ax.set_ylabel("Reflux ratio RR", color="#2980b9", fontsize=11)
    ax2.set_ylabel("Flow setpoints", color="#7f8c8d", fontsize=11)
    ax.set_title("Setpoint trajectories along the Pareto front",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.suptitle("Axion AI — Multi-Objective Optimization (NSGA-II)",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    out_path = RESULTS_DIR / "optimization_pareto.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"   Figure saved: {out_path}")

    print("\n" + "=" * 72)
    print("Optimization complete.")


if __name__ == "__main__":
    main()
