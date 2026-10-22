"""Exercise Stop against a real optimization worker and restart afterward."""
from playwright.sync_api import expect
import browser_optimization as harness
from browser_picker_helpers import choose_path


def exercise(context, address, output):
    page = context.new_page()
    errors, applied = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def capture(response):
        if "_dash-update-component" in response.url and response.status == 200:
            try:
                body = response.json().get("response", {})
            except Exception:
                return  # A final polling response can outlive browser teardown.
            if body.get("opt-applied-result", {}).get("data"):
                applied.append(body["opt-applied-result"]["data"])

    page.on("response", capture)
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")
    start, stop = page.locator("#opt-start"), page.locator("#opt-stop")
    expect(stop).to_be_disabled()
    page.locator("#opt-panel").evaluate("element => { element.open = true; }")
    page.get_by_role("tab", name="Design variables", exact=True).click()
    choose_path(page, "variable", "blade_velocity_ratio")
    start.click()
    expect(stop).to_be_enabled()
    expect(start).to_be_disabled()
    stop.click()
    expect(page.locator("#opt-state-label")).to_have_text("Stopped by user", timeout=15000)
    expect(start).to_be_enabled()
    expect(stop).to_be_disabled()
    assert not applied, "A cancelled run must not replace the current design"
    for theme in ("dark", "light"):
        page.locator(f'.td-theme-toggle [data-theme="{theme}"]').click()
        page.screenshot(path=str(output / f"optimization_stopped_{theme}.png"))
    start.click()
    expect(page.locator("#opt-state-label")).to_have_text("Applied successfully", timeout=60000)
    expect(stop).to_be_disabled()
    expect(start).to_be_enabled()
    assert applied
    assert not errors, errors
    page.remove_listener("response", capture)
    return dict(passed=True, cancelled=True, restarted=True, errors=errors)


if __name__ == "__main__":
    harness.exercise = exercise
    harness.main()
