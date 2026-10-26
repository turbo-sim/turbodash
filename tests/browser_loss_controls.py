"""Real-browser coverage of the exit Mach correction and YAML round trips."""

from copy import deepcopy
import time

import numpy as np
from playwright.sync_api import expect
import yaml

import browser_optimization as harness


def exercise(context, address, output):
    page = context.new_page()
    errors, results = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def capture(response):
        if "_dash-update-component" not in response.url:
            return
        if response.status >= 400:
            errors.append(f"Callback HTTP {response.status}")
        if response.status == 200:
            try:
                body = response.json().get("response", {})
            except Exception:
                return  # The browser may close while a background poll finishes.
            load_error = body.get("td-load-error", {}).get("children")
            if load_error:
                errors.append(load_error)
            if "result_store" in body:
                results.append(body["result_store"]["data"])

    def wait_result(enabled, model="benner", *, after=0):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            matches = [
                r
                for r in results[after:]
                if r["inputs"].get("exit_mach_correction", False) is enabled
                and r["inputs"]["loss_model"] == model
            ]
            if matches:
                return matches[-1]
            page.wait_for_timeout(100)
        raise AssertionError(
            f"No solved result for {model}, correction={enabled}; errors={errors}"
        )

    page.on("response", capture)
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")
    baseline = wait_result(False)
    page.locator("summary").filter(has_text="Turbine setup").click()
    checkbox = page.locator('#td-exit-mach-option input[type="checkbox"]')
    expect(checkbox).to_be_visible()
    expect(checkbox).not_to_be_checked()
    help_icon = page.locator("#td-exit-mach-help")
    help_icon.hover()
    assert "method of characteristics" in help_icon.get_attribute("title")
    assert "Leave unchecked" in help_icon.get_attribute("title")
    assert (
        help_icon.evaluate("element => getComputedStyle(element, '::after').display")
        == "block"
    )
    page.screenshot(path=str(output / "exit_mach_tooltip.png"))
    page.mouse.move(1000, 500)

    count = len(results)
    checkbox.check()
    enabled = wait_result(True, after=count)
    for row, index, key in (("stator", 1, "Ma_abs"), ("rotor", 3, "Ma_rel")):
        before = baseline["stages_performance"][0]
        after = enabled["stages_performance"][0]
        mach = after["flow_stations"][index][key]
        assert mach > 1
        np.testing.assert_allclose(
            after["losses"][row]["loss_profile"],
            before["losses"][row]["loss_profile"] * (1 + 60 * (mach - 1) ** 2),
        )

    with page.expect_download() as download:
        page.locator("#save_button").click()
    path = output / "correction_enabled.yaml"
    download.value.save_as(path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["inputs"]["exit_mach_correction"] is True
    count = len(results)
    checkbox.uncheck()
    disabled = wait_result(False, after=count)
    assert disabled["overall_performance"] == baseline["overall_performance"]
    count = len(results)
    page.locator('#load_button input[type="file"]').set_input_files(path)
    wait_result(True, after=count)
    expect(checkbox).to_be_checked()
    page.locator("summary").filter(has_text="Turbine setup").click()

    model = page.locator('[id=\'{"key":"loss_model","scope":"overall"}\']')
    for label, value in (
        ("Isentropic", "isentropic"),
        ("Kacker Okapuu", "kacker_okapuu"),
        ("Moustapha", "moustapha"),
        ("Benner", "benner"),
    ):
        count = len(results)
        model.click()
        page.get_by_text(label, exact=True).last.click()
        wait_result(True, value, after=count)
        if value == "isentropic":
            expect(checkbox).not_to_be_visible()
        else:
            expect(checkbox).to_be_visible()
            expect(checkbox).to_be_checked()

    legacy = deepcopy(saved)
    del legacy["inputs"]["exit_mach_correction"]
    count = len(results)
    page.locator('#load_button input[type="file"]').set_input_files(
        {
            "name": "legacy.yaml",
            "mimeType": "application/yaml",
            "buffer": yaml.safe_dump(legacy).encode(),
        }
    )
    wait_result(False, after=count)
    expect(checkbox).not_to_be_checked()
    assert not errors, errors
    return dict(
        passed=True,
        checked_models=3,
        isentropic_hidden=True,
        yaml_roundtrip=True,
        legacy_default=False,
        numerical_profile_scaling=True,
        errors=errors,
    )


if __name__ == "__main__":
    harness.exercise = exercise
    harness.main()
