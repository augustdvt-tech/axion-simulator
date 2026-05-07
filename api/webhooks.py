"""
Axion AI — Webhook notifier
============================

Posts a JSON payload to a configurable URL whenever a recommendation crosses
a configurable urgency threshold. Designed to integrate with Slack, Teams or
any inbound webhook without depending on their SDK.

Configuration via environment variables:
  AXION_WEBHOOK_URL          — destination URL (required to enable)
  AXION_WEBHOOK_URGENCY      — minimum urgency to fire (default: critical)
                               one of: low, medium, high, critical
  AXION_WEBHOOK_TIMEOUT      — POST timeout in seconds (default: 5.0)
  AXION_WEBHOOK_FORMAT       — payload format: "axion" (default) or "slack"
                               slack format wraps the message in {"text": "..."}

Design notes
------------
- The notifier swallows all network errors so it never breaks the request that
  triggered it. Failures are logged but never raised.
- Deduplication by recommendation id: a given recommendation only fires once
  per process lifetime, even if the engine re-emits it.
- The notifier is synchronous on a background thread (requests.post). For an
  MVP scale of a handful of critical recs/day this is more than enough.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional, Set
from urllib import request as urllib_request
from urllib.error import URLError

from axion_logging import get_logger

logger = get_logger(__name__)


_URGENCY_LEVEL = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _meets_threshold(urgency: str, threshold: str) -> bool:
    return _URGENCY_LEVEL.get(urgency, -1) >= _URGENCY_LEVEL.get(threshold, 99)


def _format_payload(rec: Dict[str, Any], scenario: str, fmt: str) -> Dict[str, Any]:
    """Build the body to POST. Slack format wraps in {"text": "..."}."""
    if fmt == "slack":
        msg = (
            f":rotating_light: *AXION AI — {rec.get('urgency', '').upper()}*\n"
            f"*Scenario*: {scenario}\n"
            f"*Rule*: {rec.get('rule_fired', '—')}\n"
            f"*Diagnosis*: {rec.get('diagnosis', '—')}\n"
            f"*Action*: {rec.get('action', '—')}"
        )
        return {"text": msg}
    # Default native Axion format — full structured payload
    return {
        "event":     "recommendation",
        "scenario":  scenario,
        "rec_id":    rec.get("id"),
        "timestamp": rec.get("timestamp"),
        "urgency":   rec.get("urgency"),
        "rule":      rec.get("rule_fired"),
        "diagnosis": rec.get("diagnosis"),
        "action":    rec.get("action"),
    }


def _post(url: str, body: Dict[str, Any], timeout: float) -> bool:
    """POST JSON to the URL. Returns True on success, False on failure."""
    data = json.dumps(body).encode("utf-8")
    req = urllib_request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (URLError, TimeoutError) as e:
        logger.warning("Webhook POST failed",
                       extra={"url": url, "error": str(e)})
        return False
    except Exception as e:
        logger.error("Webhook POST raised unexpected error",
                     extra={"url": url, "error": str(e)})
        return False


class WebhookNotifier:
    """Fires webhooks for recommendations above a configured urgency threshold.

    Stateful — keeps a set of already-notified rec ids so the same recommendation
    is never fired twice.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        threshold: str = "critical",
        timeout: float = 5.0,
        fmt: str = "axion",
        async_post: bool = True,
    ):
        self.url = url
        self.threshold = threshold
        self.timeout = timeout
        self.fmt = fmt
        self.async_post = async_post
        self._fired: Set[str] = set()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    @classmethod
    def from_env(cls) -> "WebhookNotifier":
        return cls(
            url=os.environ.get("AXION_WEBHOOK_URL", "").strip() or None,
            threshold=os.environ.get("AXION_WEBHOOK_URGENCY", "critical").strip().lower(),
            timeout=float(os.environ.get("AXION_WEBHOOK_TIMEOUT", "5.0")),
            fmt=os.environ.get("AXION_WEBHOOK_FORMAT", "axion").strip().lower(),
        )

    def should_fire(self, rec: Dict[str, Any]) -> bool:
        """Return True if this rec passes the threshold and hasn't fired yet."""
        if not self.enabled:
            return False
        urgency = (rec.get("urgency") or "").lower()
        if not _meets_threshold(urgency, self.threshold):
            return False
        rec_id = rec.get("id")
        if rec_id is None:
            return True   # no id ⇒ can't dedupe; let it fire
        with self._lock:
            return rec_id not in self._fired

    def notify(self, rec: Dict[str, Any], scenario: str) -> bool:
        """Fire the webhook for this recommendation. Returns True on success.

        Safe to call unconditionally — it self-checks should_fire() and dedupes.
        """
        if not self.should_fire(rec):
            return False
        rec_id = rec.get("id")
        if rec_id is not None:
            with self._lock:
                self._fired.add(rec_id)

        body = _format_payload(rec, scenario, self.fmt)

        if self.async_post:
            t = threading.Thread(
                target=_post, args=(self.url, body, self.timeout),
                daemon=True,
            )
            t.start()
            return True
        return _post(self.url, body, self.timeout)

    def reset(self) -> None:
        """Forget the dedupe set — useful when scenario reloads."""
        with self._lock:
            self._fired.clear()
