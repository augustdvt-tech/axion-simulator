"""Tests for axion_logging.py: setup_logging, get_logger, formatters."""

import json
import logging
import os
import pytest


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset axion_logging state and root logger between tests."""
    import axion_logging
    axion_logging._configured = False
    root = logging.getLogger()
    root.handlers.clear()
    yield
    axion_logging._configured = False
    root.handlers.clear()


class TestSetupLogging:
    def test_setup_logging_idempotent(self):
        import axion_logging
        axion_logging.setup_logging(level="INFO", fmt="pretty")
        axion_logging.setup_logging(level="DEBUG", fmt="json")   # second call is no-op
        assert logging.getLogger().level == logging.INFO   # first call wins

    def test_setup_logging_sets_level_info(self):
        import axion_logging
        axion_logging.setup_logging(level="INFO", fmt="pretty")
        assert logging.getLogger().level == logging.INFO

    def test_setup_logging_sets_level_debug(self):
        import axion_logging
        axion_logging.setup_logging(level="DEBUG", fmt="pretty")
        assert logging.getLogger().level == logging.DEBUG

    def test_setup_logging_sets_level_warning(self):
        import axion_logging
        axion_logging.setup_logging(level="WARNING", fmt="pretty")
        assert logging.getLogger().level == logging.WARNING

    def test_env_var_log_level(self, monkeypatch):
        monkeypatch.setenv("AXION_LOG_LEVEL", "ERROR")
        import axion_logging
        axion_logging.setup_logging()
        assert logging.getLogger().level == logging.ERROR

    def test_env_var_log_format_json(self, monkeypatch, capsys):
        monkeypatch.setenv("AXION_LOG_FORMAT", "json")
        import axion_logging
        axion_logging.setup_logging()
        logger = axion_logging.get_logger("test.json")
        logger.info("hello json", extra={"key": "value"})
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert obj["message"] == "hello json"
        assert obj["level"] == "INFO"


class TestGetLogger:
    def test_returns_logger_instance(self):
        import axion_logging
        log = axion_logging.get_logger("axion.test")
        assert isinstance(log, logging.Logger)

    def test_logger_name(self):
        import axion_logging
        log = axion_logging.get_logger("axion.mymodule")
        assert log.name == "axion.mymodule"

    def test_get_logger_triggers_setup(self):
        import axion_logging
        axion_logging.get_logger("x")
        assert axion_logging._configured is True


class TestPrettyFormatter:
    def test_extra_fields_appear_in_output(self, capsys):
        import axion_logging
        axion_logging.setup_logging(level="DEBUG", fmt="pretty")
        logger = axion_logging.get_logger("test.pretty")
        logger.info("startup", extra={"scenario": "normal", "samples": 100})
        captured = capsys.readouterr()
        assert "scenario=normal" in captured.out
        assert "samples=100" in captured.out

    def test_no_extra_fields_clean_output(self, capsys):
        import axion_logging
        axion_logging.setup_logging(level="INFO", fmt="pretty")
        logger = axion_logging.get_logger("test.clean")
        logger.info("clean message")
        captured = capsys.readouterr()
        assert "clean message" in captured.out
        assert "[" not in captured.out   # no extra block


class TestJsonFormatter:
    def test_json_output_valid(self, capsys):
        import axion_logging
        axion_logging.setup_logging(level="INFO", fmt="json")
        logger = axion_logging.get_logger("test.json2")
        logger.warning("something happened", extra={"code": 42})
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert obj["level"] == "WARNING"
        assert obj["message"] == "something happened"
        assert obj["code"] == 42

    def test_json_has_timestamp(self, capsys):
        import axion_logging
        axion_logging.setup_logging(level="INFO", fmt="json")
        logger = axion_logging.get_logger("test.ts")
        logger.info("ts test")
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert "timestamp" in obj
        assert "T" in obj["timestamp"]   # ISO 8601 format
