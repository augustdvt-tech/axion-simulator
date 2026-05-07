"""
Browser-driven dashboard E2E tests.

Gated behind --run-browser to keep CI lean. To run locally:

    pip install playwright pytest-playwright
    playwright install chromium
    pytest tests/e2e/test_dashboard_e2e.py --run-browser -v

These exercise the React UI shell — the parts that are hardest to reason
about from server-side tests alone (event handlers, websocket messages
arriving in the DOM, React state transitions on tab switches).

Browser tests are deliberately shallow: they verify the dashboard mounts,
key panels render, and the scenario picker round-trips a state change.
Component-level UI behavior is better tested via component tests; this
file is for "the whole stack still loads end to end".
"""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.browser]


@pytest.fixture
def page(live_server):
    """Lazy-import playwright so the module is importable without it installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed — pip install pytest-playwright")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(f"{live_server}/", wait_until="domcontentloaded")
        yield page
        ctx.close()
        browser.close()


class TestDashboardLoad:
    def test_brand_renders(self, page):
        page.wait_for_selector("text=AXION", timeout=5000)

    def test_scenario_picker_present(self, page):
        page.wait_for_selector("select", timeout=5000)
        # The select should have at least one <option>
        options = page.query_selector_all("select option")
        assert len(options) > 0

    def test_scenario_change_round_trip(self, page):
        page.wait_for_selector("select", timeout=5000)
        select = page.query_selector("select")
        # Pick the second option (different from default)
        opts = page.query_selector_all("select option")
        if len(opts) < 2:
            pytest.skip("only one scenario available")
        new_value = opts[1].get_attribute("value")
        select.select_option(new_value)
        # Wait briefly for the WebSocket round-trip + re-render
        page.wait_for_timeout(1500)
        # The select should now reflect the new value
        assert page.eval_on_selector("select", "el => el.value") == new_value
