"""
Axion AI - Industrial Integration Layer
=======================================

OPC-UA source and writer clients for connecting Axion AI to real plant DCS/PLC
systems. Plus a mock server for testing the client without a real plant.

Primary components:
- TagMap:          Declarative mapping from plant node IDs to Axion tags
- OPCUASource:     Reads values from OPC-UA server, emits Sample stream
- OPCUAWriter:     Writes setpoints with safety gates + readback verification
- IngestionService: Buffers samples, runs analytics pipeline periodically
- opcua_mock_server: Standalone OPC-UA server that replays a scenario CSV
"""

from .tag_map import (
    TagMap, TagMapping, ServerConfig, SamplingConfig, PILOT_TAG_MAP,
)
from .opcua_source import OPCUASource, Sample
from .opcua_writer import OPCUAWriter, WriteResult, WriteStatus
from .ingestion import IngestionService, IngestionConfig

__all__ = [
    "TagMap", "TagMapping", "ServerConfig", "SamplingConfig", "PILOT_TAG_MAP",
    "OPCUASource", "Sample",
    "OPCUAWriter", "WriteResult", "WriteStatus",
    "IngestionService", "IngestionConfig",
]
