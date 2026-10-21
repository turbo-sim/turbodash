"""Browser regression for native SLSQP controls and physical-domain recovery.

Run from the repository root with ``python tests/browser_slsqp_recovery.py``.
Accepts the same --chromium and --output arguments as browser_optimization.py.
"""

from copy import deepcopy
import time

import yaml
from playwright.sync_api import expect

import browser_optimization as harness
from turbodash.apps.app_turbine_optimization.opt_model import (
    SLSQP_EXTRA_CONTROLS, solver_editor_options,
)


def exercise(context, address, output):
    page = context.new_page()
    errors, results, applied = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def capture(response):
        if "_dash-update-component" not in response.url:
            return
        if response.status >= 400:
            errors.append(f"Callback HTTP {response.status}: {response.request.post_data[:160]}")
        if response.status == 200:
            try:
                body = response.json().get("response", {})
            except Exception:
                return
            if "result_store" in body:
                results.append(body["result_store"]["data"])
            if "opt-applied-result" in body and body["opt-applied-result"].get("data"):
                applied.append(body["opt-applied-result"]["data"])

    page.on("response", capture)
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")
    source = yaml.safe_load((harness.ROOT / "demos/optimization/case_recovery.yaml").read_text())
    # Preserve the original recovery regression trajectory explicitly.
    source["optimization"]["solver_options"]["step_growth_factor"] = None
    assert len(source["optimization"]["design_variables"]) == 14

    def open_section(identity):
        section = page.locator(f"#{identity}")
        if section.get_attribute("open") is None:
            section.locator(":scope > summary").click()
        expect(section).to_have_js_property("open", True)

    def upload(config, filename):
        previous = len(results)
        page.locator('#load_button input[type="file"]').set_input_files({
            "name": filename, "mimeType": "text/yaml", "buffer": yaml.safe_dump(config).encode(),
        })
        deadline = time.monotonic() + 30
        while len(results) <= previous or abs(
            results[-1]["inputs"]["blade_velocity_ratio"] - config["inputs"]["blade_velocity_ratio"]
        ) > 1e-12:
            if time.monotonic() > deadline:
                raise AssertionError("Upload did not calculate: " + page.locator("#td-load-error").inner_text())
            page.wait_for_timeout(100)
        open_section("opt-panel")
        page.get_by_role("tab", name="Configuration", exact=True).click()
        page.locator("#opt-expert-settings").check()
        options = solver_editor_options(config["optimization"])
        for key, identity in SLSQP_EXTRA_CONTROLS:
            control = page.locator(f"#{identity}")
            if key == "domain_backtracking":
                expect(control).to_be_checked(checked=options[key])
            elif options[key] is None:
                expect(control).to_have_value("")
            else:
                expect(control).to_have_js_property("valueAsNumber", options[key])
        expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(14)

    upload(source, "all-variables-flaring.yaml")
    exported_options = solver_editor_options(source["optimization"])
    for identity, value in (("opt-first-step", .75), ("opt-step-growth-factor", 1.5),
                            ("opt-backtrack-factor", .5), ("opt-max-backtracks", 15)):
        page.locator(f"#{identity}").fill(str(value))
        page.locator(f"#{identity}").blur()
    exported_options.update(first_step_fraction=.75, step_growth_factor=1.5,
                            backtrack_factor=.5, max_backtracks=15)
    page.locator("#opt-panel").screenshot(path=output / "slsqp-controls.png")
    expect(page.locator("#save_button")).to_be_enabled()
    with page.expect_download() as download:
        page.locator("#save_button").click()
    saved_path = output / "slsqp-controls-export.yaml"
    download.value.save_as(saved_path)
    exported = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    assert exported["optimization"]["solver_options"] == exported_options
    assert exported["optimization"]["feasibility_tolerance"] == 1e-6
    assert "feasibility_tol" not in exported["optimization"]["solver_options"]
    upload(exported, "slsqp-controls-roundtrip.yaml")
    page.locator("#opt-first-step").fill("1")
    page.locator("#opt-first-step").blur()
    page.locator("#opt-step-growth-factor").fill("")
    page.locator("#opt-step-growth-factor").blur()

    def run():
        page.locator("#opt-start").click()
        expect(page.locator("#opt-start")).to_be_disabled(timeout=10000)
        expect(page.locator("#opt-start")).to_be_enabled(timeout=120000)
        expect(page.locator("#opt-panel")).to_have_js_property("open", True)
        expect(page.locator("#opt-tabs .nav-link.active")).to_have_text("Configuration")
        return page.locator("#opt-status").inner_text()

    success_status = run()
    assert "Optimization applied" in success_status, success_status
    assert "Recovered 1 invalid trial" in success_status, success_status
    expect(page.locator("#opt-log")).to_contain_text("Success: True")
    expect(page.locator("#opt-log")).to_contain_text("Rejected trial")
    expect(page.locator("#opt-log")).to_contain_text("rotor_exit_radicand")
    for metric in ("Step-norm", "Optimality", "Complement.", "KKT/tol", "Final KKT:"):
        expect(page.locator("#opt-log")).to_contain_text(metric)
    successful = deepcopy(applied[-1])
    assert successful["diagnostics_available"]
    assert successful["kkt"]["passed"]
    assert successful["kkt_ratio"] <= 1.0
    assert successful["history"][-1]["optimality"] == successful["optimality"]
    assert successful["domain_rejections"] == 1
    assert abs(100 * successful["objective"] - 89.811945) < 1e-5
    assert results[-1]["inputs"]["stages"] == successful["config"]["inputs"]["stages"]
    assert harness.flat_speed(results[-1]["inputs"]) == harness.flat_speed(successful["config"]["inputs"])
    rejected = next(event for event in successful["recovery_events"] if event["kind"] == "trial_rejection")
    assert rejected["failure"]["quantity"] == "rotor_exit_radicand"
    assert rejected["failure"]["margin"] < 0
    assert rejected["failure"]["stage_index"] == 0
    assert rejected["failure"]["station"] == 4
    page.locator("#opt-terminal").scroll_into_view_if_needed()
    page.screenshot(path=output / "slsqp-recovered-success.png")
    (output / "successful-solver.log").write_text(page.locator("#opt-log").inner_text(), encoding="utf-8")

    # Recreate the original failed first step with recovery explicitly disabled.
    failed_case = deepcopy(source)
    failed_case["optimization"]["solver_options"]["domain_backtracking"] = False
    upload(failed_case, "all-variables-without-recovery.yaml")
    before_inputs, before_applied = deepcopy(results[-1]["inputs"]), len(applied)
    failure_status = run()
    assert "Result not applied" in failure_status, failure_status
    expect(page.locator("#opt-log")).to_contain_text("Success: False")
    expect(page.locator("#opt-log")).to_contain_text("rotor_exit_radicand")
    assert results[-1]["inputs"] == before_inputs
    assert len(applied) == before_applied
    expect(page.locator('input[id*=blade_velocity_ratio]')).to_have_js_property(
        "valueAsNumber", source["inputs"]["blade_velocity_ratio"],
    )
    expect(page.locator("#opt-state-label")).to_have_text("Optimization failed")
    expect(page.locator("#opt-start")).to_have_class("td-opt-start btn btn-danger")
    (output / "failed-solver.log").write_text(page.locator("#opt-log").inner_text(), encoding="utf-8")
    page.screenshot(path=output / "slsqp-recovery-disabled.png")
    assert not errors, errors
    return dict(
        checks=["all 14 original variables and flaring constraints loaded", "native control defaults",
                "live advanced controls exported and reloaded without changes", "adaptive recovery of invalid rotor exit",
                "optimized design applied with physical efficiency", "structured rejection metadata retained",
                "native rejection diagnosis visible in terminal", "optimization panel and tab retained",
                "step and same-point KKT metrics displayed and persisted",
                "disabled recovery fails cleanly and preserves original inputs"],
        successful_efficiency_percent=100 * successful["objective"],
        domain_rejections=successful["domain_rejections"], rejected_failure=rejected["failure"],
        kkt_ratio=successful["kkt_ratio"], optimality=successful["optimality"],
        success_status=success_status, disabled_recovery_status=failure_status,
        exported_solver_options=exported_options, browser_errors=errors,
    )


if __name__ == "__main__":
    harness.exercise = exercise
    harness.main()
