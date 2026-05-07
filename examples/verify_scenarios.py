"""
Scenario verification: shows that each scenario produces distinct, expected
dynamics. This is our smoke test that the simulator is working correctly.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "data"

scenarios = [
    "normal", "thermal_drift", "feed_perturbation", "reactor_instability",
    "quality_degradation", "energy_waste", "product_grade_change", "sensor_failure",
]

fig, axes = plt.subplots(4, 2, figsize=(16, 14))
axes = axes.flatten()

expectations = {
    "normal": "Baseline: stable around setpoints with measurement noise",
    "thermal_drift": "T_R rises slowly as jacket fouling degrades heat removal",
    "feed_perturbation": "Step at t=12h in feed concentration propagates through both units",
    "reactor_instability": "Oscillations in T_R from sinusoidal coolant disturbance",
    "quality_degradation": "Product purity degrades as column volatility drops",
    "energy_waste": "Step increase in reflux at t=4h: purity unchanged, energy up",
    "product_grade_change": "Step setpoint change at t=10h pushes purity to new level",
    "sensor_failure": "T_R sensor frozen between t=8h and t=12h",
}

for i, scn in enumerate(scenarios):
    ax = axes[i]
    df = pd.read_csv(DATA_DIR / f"{scn}.csv")
    t_h = df["time_s"] / 3600.0

    # For each scenario, show the most informative variable(s)
    if scn == "normal":
        ax.plot(t_h, df["cstr.T_R_C"], label="T_R", color="#c0392b", lw=1)
        ax.plot(t_h, df["column.purity_B"], label="Purity B", color="#2980b9", lw=1)
        ax.set_ylabel("T_R [°C]  |  Purity [%]")
    elif scn == "thermal_drift":
        ax.plot(t_h, df["cstr.T_R_C"], label="T_R", color="#c0392b", lw=1)
        ax.plot(t_h, df["cstr.T_J_C"], label="T_J", color="#3498db", lw=1)
        ax.set_ylabel("Temperature [°C]")
    elif scn == "feed_perturbation":
        ax.plot(t_h, df["cstr.C_A"], label="C_A", color="#27ae60", lw=1)
        ax2 = ax.twinx()
        ax2.plot(t_h, df["column.purity_B"], label="Purity B", color="#2980b9", lw=1)
        ax2.set_ylabel("Purity [%]", color="#2980b9")
        ax.set_ylabel("C_A [mol/m³]", color="#27ae60")
    elif scn == "reactor_instability":
        ax.plot(t_h, df["cstr.T_R_C"], label="T_R", color="#c0392b", lw=0.8)
        ax.plot(t_h, df["cstr.F_cool"] * 80, label="F_cool × 80", color="#3498db", lw=0.8, alpha=0.7)
        ax.set_ylabel("T_R [°C]  |  F_cool × 80")
    elif scn == "quality_degradation":
        ax.plot(t_h, df["column.purity_B"], label="Purity B", color="#2980b9", lw=1)
        ax.axhline(98.5, color="red", linestyle="--", lw=1, alpha=0.6, label="Spec ≥ 98.5%")
        ax.set_ylabel("Purity B [%]")
    elif scn == "energy_waste":
        ax.plot(t_h, df["column.RR"], label="RR", color="#9b59b6", lw=1)
        ax2 = ax.twinx()
        ax2.plot(t_h, df["column.Q_reb_kW"], label="Q_reb", color="#e67e22", lw=1)
        ax2.set_ylabel("Q_reb [kW]", color="#e67e22")
        ax.set_ylabel("Reflux Ratio", color="#9b59b6")
    elif scn == "product_grade_change":
        ax.plot(t_h, df["column.RR"], label="RR setpoint", color="#9b59b6", lw=1)
        ax2 = ax.twinx()
        ax2.plot(t_h, df["column.purity_B"], label="Purity B", color="#2980b9", lw=1)
        ax2.set_ylabel("Purity B [%]", color="#2980b9")
        ax.set_ylabel("Reflux Ratio", color="#9b59b6")
    elif scn == "sensor_failure":
        ax.plot(t_h, df["cstr.T_R_C"], label="T_R (measured)", color="#c0392b", lw=1)
        ax.axvspan(8, 12, alpha=0.2, color="orange", label="Sensor failure window")
        ax.set_ylabel("T_R [°C]")

    ax.set_title(f"{scn}\n{expectations[scn]}", fontsize=9)
    ax.set_xlabel("Time [h]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

plt.suptitle("Axion AI Simulator — Scenario Verification", fontsize=14, y=1.00)
plt.tight_layout()
out_path = FIG_DIR / "scenarios_verification.png"
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print(f"Figure saved to: {out_path}")

# Print statistics summary
print("\n" + "=" * 80)
print(f"{'SCENARIO':<25} {'T_R range':<18} {'Purity range':<18} {'Q_reb range':<15}")
print("=" * 80)
for scn in scenarios:
    df = pd.read_csv(DATA_DIR / f"{scn}.csv")
    t_r = df["cstr.T_R_C"]
    pur = df["column.purity_B"]
    q   = df["column.Q_reb_kW"]
    print(f"{scn:<25} {t_r.min():6.2f} → {t_r.max():6.2f}    {pur.min():6.2f} → {pur.max():6.2f}   "
          f"{q.min():5.1f} → {q.max():5.1f}")
