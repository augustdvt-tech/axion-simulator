"""
Axion AI Process Simulator
==========================

A modular, scalable simulator for industrial processes. Currently implements
a CSTR + binary distillation column pilot; extensible to any process by
subclassing ProcessUnit.
"""

from .core import ProcessUnit, Stream, Simulator, sequential_connections
from .units import CSTR, DistillationColumn
from .scenarios import (
    Scenario, CompositeScenario, NormalOperation, ThermalDrift,
    FeedPerturbation, ReactorInstability, QualityDegradation,
    EnergyWaste, SensorFailureScenario, ProductGradeChange,
    SCENARIO_REGISTRY,
)
from .instrumentation import SensorModel, DataLogger

__version__ = "0.1.0"

__all__ = [
    "ProcessUnit", "Stream", "Simulator", "sequential_connections",
    "CSTR", "DistillationColumn",
    "Scenario", "CompositeScenario", "NormalOperation", "ThermalDrift",
    "FeedPerturbation", "ReactorInstability", "QualityDegradation",
    "EnergyWaste", "SensorFailureScenario", "ProductGradeChange",
    "SCENARIO_REGISTRY",
    "SensorModel", "DataLogger",
]
