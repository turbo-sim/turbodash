"""Exercise solver selection, adaptive IPM, persistence, cancellation, and retry.

Run with the browser_optimization harness's --chromium and --output arguments.
All screenshots, exports, worker snapshots, and check results stay in output.
"""

from copy import deepcopy
import json
import time

from playwright.sync_api import expect
import yaml

import browser_optimization as harness


def exercise(context, address, output):
    """Check real editor interactions and a constrained axial-turbine worker."""
    page = context.new_page()
    page.add_init_script("window.localStorage.clear(); window.sessionStorage.clear();")
    errors, results, applied, handles, progress = [], [], [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def capture(response):
        """Retain delivered app state and callback failures without changing it."""
        if "_dash-update-component" not in response.url:
            return
        if response.status >= 400:
            errors.append(
                f"Callback HTTP {response.status}: {response.request.post_data[:160]}"
            )
        if response.status != 200:
            return
        try:
            body = response.json().get("response", {})
        except Exception:
            return
        for key, records in (
            ("result_store", results),
            ("opt-applied-result", applied),
            ("opt-progress-snapshot", progress),
        ):
            value = body.get(key, {}).get("data")
            if value:
                records.append(value)
        handle = body.get("opt-job", {}).get("data")
        if handle and (not handles or handles[-1]["id"] != handle["id"]):
            handles.append(handle)

    page.on("response", capture)
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")

    def open_section(identity):
        """Use the disclosure's actual control when opening the editor or log."""
        section = page.locator(f"#{identity}")
        if section.get_attribute("open") is None:
            section.locator(":scope > summary").click()
        expect(section).to_have_js_property("open", True)

    def select(identity, label):
        """Select one of the rendered Dash dropdown choices."""
        page.locator(f"#{identity} .Select-control").click()
        page.locator(f"#{identity} .VirtualizedSelectOption").filter(
            has_text=label
        ).click()
        expect(page.locator(f"#{identity} .Select-value-label")).to_have_text(label)

    open_section("opt-panel")
    page.get_by_role("tab", name="Configuration", exact=True).click()
    expect(page.locator("#opt-solver .Select-value-label")).to_have_text("SLSQP")
    expect(page.locator("#opt-panel > summary #opt-solver-badge")).to_have_text("SLSQP")
    expect(page.locator("#opt-terminal-solver")).to_have_text("SLSQP")
    expect(page.locator("#opt-expert-ipm")).to_be_hidden()
    page.locator("#opt-expert-settings").check()
    expect(page.locator("#opt-expert-steps")).to_be_visible()
    select("opt-solver", "IPM")
    expect(page.locator("#opt-expert-ipm")).to_be_visible()
    expect(page.locator("#opt-expert-steps")).to_be_hidden()
    expect(page.locator("#opt-feasibility")).to_have_js_property("valueAsNumber", 1e-6)
    expect(page.locator("#opt-mu-strategy .Select-value-label")).to_have_text(
        "Adaptive"
    )
    select("opt-mu-strategy", "Monotone")
    select("opt-solver", "SLSQP")
    expect(page.locator("#opt-solver-badge")).to_have_text("SLSQP")
    expect(page.locator("#opt-expert-steps")).to_be_visible()
    expect(page.locator("#opt-expert-ipm")).to_be_hidden()
    for identity, value in (("opt-first-step", 0.37), ("opt-step-growth-factor", 1.7)):
        page.locator(f"#{identity}").fill(str(value))
        page.locator(f"#{identity}").blur()
    page.locator("#opt-panel").screenshot(path=output / "slsqp-expert-controls.png")
    select("opt-solver", "SQP")
    expect(page.locator("#opt-solver-badge")).to_have_text("SQP")
    expect(page.locator("#opt-expert-steps")).to_be_hidden()
    expect(page.locator("#opt-expert-ipm")).to_be_hidden()
    expect(page.locator("#opt-expert-tolerances")).to_be_visible()
    select("opt-solver", "SLSQP")
    expect(page.locator("#opt-first-step")).to_have_js_property("valueAsNumber", 0.37)
    expect(page.locator("#opt-step-growth-factor")).to_have_js_property(
        "valueAsNumber", 1.7
    )
    select("opt-solver", "IPM")
    expect(page.locator("#opt-mu-strategy .Select-value-label")).to_have_text(
        "Monotone"
    )
    page.locator("#opt-panel").screenshot(path=output / "ipm-expert-controls.png")

    source = yaml.safe_load(
        (harness.ROOT / "demos/optimization/case_axial.yaml").read_text()
    )
    source["optimization"]["solver"] = "pounce_ipm"
    source["optimization"]["feasibility_tolerance"] = 1e-6
    source["optimization"]["solver_options"] = dict(
        ftol=1e-8,
        maxiter=150,
        finite_diff_step=1e-6,
        optimality_tol=1e-6,
        mu_strategy="monotone",
    )

    def upload(config, name):
        """Wait for both the physical result and optimizer settings to reload."""
        before = len(results)
        page.locator('#load_button input[type="file"]').set_input_files(
            {
                "name": name,
                "mimeType": "text/yaml",
                "buffer": yaml.safe_dump(config).encode(),
            }
        )
        deadline = time.monotonic() + 30
        while len(results) <= before:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "Upload did not calculate: "
                    + page.locator("#td-load-error").inner_text()
                )
            page.wait_for_timeout(100)
        open_section("opt-panel")
        page.get_by_role("tab", name="Configuration", exact=True).click()
        expect(page.locator("#opt-solver .Select-value-label")).to_have_text(
            "IPM"
        )
        expect(page.locator("#opt-mu-strategy .Select-value-label")).to_have_text(
            "Monotone"
        )
        expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(12)

    upload(source, "axial-ipm.yaml")
    expect(page.locator("#save_button")).to_be_enabled()
    with page.expect_download() as download:
        page.locator("#save_button").click()
    exported_path = output / "ipm-optimizer-roundtrip.yaml"
    download.value.save_as(exported_path)
    exported = yaml.safe_load(exported_path.read_text(encoding="utf-8"))
    assert exported["optimization"]["solver"] == "pounce_ipm"
    assert (
        exported["optimization"]["solver_options"]
        == source["optimization"]["solver_options"]
    )
    assert exported["optimization"]["feasibility_tolerance"] == 1e-6
    upload(exported, "ipm-roundtrip.yaml")
    select("opt-mu-strategy", "Adaptive")

    start, stop = page.locator("#opt-start"), page.locator("#opt-stop")
    original_inputs = deepcopy(results[-1]["inputs"])
    start.click()
    expect(stop).to_be_enabled(timeout=10000)
    expect(start).to_be_disabled()
    expect(page.locator('#opt-solver [role="combobox"]')).to_be_disabled()
    expect(page.locator("#opt-terminal-solver")).to_have_text("IPM")
    expect(page.locator("#opt-log")).to_contain_text(
        "Optimizer: IPM", timeout=30000
    )
    stop.click()
    expect(page.locator("#opt-state-label")).to_have_text(
        "Stopped by user", timeout=15000
    )
    expect(start).to_be_enabled()
    expect(stop).to_be_disabled()
    assert not applied, "Cancellation must not apply an optimized design"
    assert results[-1]["inputs"] == original_inputs
    page.screenshot(path=output / "ipm-cancelled.png")

    start.click()
    expect(stop).to_be_enabled(timeout=10000)
    expect(page.locator("#opt-state-label")).to_have_text(
        "Applied successfully", timeout=120000
    )
    expect(start).to_be_enabled()
    expect(stop).to_be_disabled()
    assert applied
    successful = applied[-1]
    assert successful["solver"] == "pounce_ipm"
    assert successful["success"] and successful["kkt"]["passed"]
    assert successful["kkt_ratio"] <= 1
    assert successful["stopped_on_kkt"]
    assert successful["solver_status"] == 5
    assert abs(100 * successful["objective"] - 81.54976169) < 2e-5
    assert results[-1]["inputs"]["stages"] == successful["config"]["inputs"]["stages"]
    assert harness.flat_speed(results[-1]["inputs"]) == harness.flat_speed(
        successful["config"]["inputs"]
    )
    assert any(
        row.get("event_kind") == "native_iteration" for row in successful["history"]
    )
    request_path = (
        output / "temp/turbodash_optimization_jobs" / handles[-1]["id"] / "request.json"
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["optimization"]["solver"] == "pounce_ipm"
    requested_options = request["optimization"]["solver_options"]
    assert requested_options["mu_strategy"] == "adaptive"
    assert requested_options["optimality_tol"] == 1e-6
    assert (
        not {"first_step_fraction", "step_growth_factor", "domain_backtracking"}
        & requested_options.keys()
    )
    open_section("opt-terminal")
    expect(page.locator("#opt-log")).to_contain_text("verified user-KKT stop: True")
    page.screenshot(path=output / "ipm-success-report.png")
    (output / "successful-ipm.log").write_text(
        page.locator("#opt-log").inner_text(), encoding="utf-8"
    )
    select("opt-solver", "SLSQP")
    expect(page.locator("#opt-solver-badge")).to_have_text("SLSQP")
    expect(page.locator("#opt-terminal-solver")).to_have_text("IPM")
    page.remove_listener("response", capture)
    assert not errors, errors
    return dict(
        passed=True,
        checks=[
            "SLSQP default and explicit IPM selection",
            "solver-dependent advanced controls",
            "hidden values retained across switches",
            "selected solver and barrier YAML roundtrip",
            "Pounce cancellation preserves design",
            "retry succeeds with actual IPM worker",
            "strict feasibility and KKT verified",
            "normalized native observations retained",
            "actual worker configuration excludes SLSQP controls",
            "terminal retains executed solver label",
        ],
        efficiency_percent=100 * successful["objective"],
        kkt_ratio=successful["kkt_ratio"],
        native_status=successful["solver_status"],
        physical_violation=successful["max_constraint_violation"],
        worker_options=requested_options,
        cancelled=True,
        restarted=True,
        errors=errors,
    )


if __name__ == "__main__":
    harness.exercise = exercise
    harness.main()
