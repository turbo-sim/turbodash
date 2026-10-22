"""Focused browser regression: python tests/browser_diffuser.py --chromium PATH.

Reuse the optimization harness's local server and browser lifecycle.
"""

import yaml
from playwright.sync_api import expect

import browser_optimization as harness
from browser_picker_helpers import open_picker


def exercise(context, address, output):
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")
    expect(page.locator("#td-metrics .td-metric")).to_have_count(5)
    section = page.locator("#td-section-diffuser")
    section.locator(":scope > summary").click()
    control = section.locator('input[type="number"]')
    expect(control).to_have_js_property("valueAsNumber", 60)
    tooltip = section.locator(".td-help").get_attribute("title")
    assert "0% recovery gives the total-to-static" in tooltip
    assert "100% gives the total-to-total" in tooltip
    assert "partial recovery" in tooltip

    def result():
        return page.evaluate("""() => {
            const state = window.store.getState();
            return state.paths.strs.result_store.reduce((node, key) => node[key], state.layout).props.data;
        }""")

    baseline = result()
    for percent in (0, 100, 60):
        control.fill(str(percent))
        control.blur()
        page.wait_for_function("""percent => {
            const state = window.store.getState();
            const result = state.paths.strs.result_store.reduce((node, key) => node[key], state.layout).props.data;
            return result.inputs.diffuser.kinetic_energy_recovery_fraction === percent / 100;
        }""", arg=percent)
        solved = result()
        performance = solved["overall_performance"]
        value = performance["efficiency_ts_with_recovery"]
        if percent == 0:
            assert value == performance["efficiency_ts"]
        if percent == 100:
            assert value == performance["efficiency_tt"]
        assert performance["power_actual"] == baseline["overall_performance"]["power_actual"]
        expect(page.locator("#td-metrics .td-metric").nth(2).locator("strong")).to_have_text(f"{100 * value:,.2f}")
        page.wait_for_function("""value => {
            const graph = document.querySelector('#efficiency_trends_plot .js-plotly-plot');
            const point = graph && graph.data.find(trace => trace.name === 'Design point');
            return point && point.y.length === 3 && Math.abs(point.y[2] - 100 * value) < 1e-8;
        }""", arg=value)
        assert page.locator('#efficiency_trends_plot .js-plotly-plot').evaluate(
            "graph => graph.data.filter(trace => trace.mode === 'lines').length === 3")
        total = page.locator('#loss_plot .js-plotly-plot').evaluate(
            "graph => graph.data.reduce((sum, trace) => sum + trace.y.reduce((a, b) => a + b, 0), 0)")
        assert abs(total - 100 * (1 - value)) < 1e-8
        energy = page.locator('#loss_plot .js-plotly-plot').evaluate(
            "graph => graph.data.find(trace => trace.meta.theme_role === 'loss_exit_kinetic').y.at(-1)")
        leaving = solved['stages_performance'][-1]['flow_stations'][3]['v']**2 / 2
        expected = 100 * (1 - percent / 100) * leaving / (
            performance['isentropic_enthalpy_drop'] - percent / 100 * leaving)
        assert abs(energy - expected) < 1e-8

    with page.expect_download() as download:
        page.locator("#save_button").click()
    path = output / "diffuser_roundtrip.yaml"
    download.value.save_as(path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = list(saved["inputs"])
    assert keys.index("diffuser") == keys.index("stages") + 1
    assert saved["inputs"]["diffuser"]["kinetic_energy_recovery_fraction"] == .6
    assert "efficiency_ts_with_recovery" in saved["overall_performance"]
    saved["inputs"]["diffuser"]["kinetic_energy_recovery_fraction"] = .75
    page.locator('#load_button input[type="file"]').set_input_files({
        "name": "diffuser.yaml", "mimeType": "application/yaml",
        "buffer": yaml.safe_dump(saved).encode(),
    })
    expect(control).to_have_js_property("valueAsNumber", 75)
    page.wait_for_function("""() => {
        const state = window.store.getState();
        const result = state.paths.strs.result_store.reduce((node, key) => node[key], state.layout).props.data;
        return result.inputs.diffuser.kinetic_energy_recovery_fraction === .75;
    }""")

    page.locator("#opt-panel > summary").click()
    page.get_by_role("tab", name="Configuration", exact=True).click()
    page.locator("#opt-objective").click()
    expect(page.get_by_text("Overall · Efficiency with exhaust recovery [-]", exact=True)).to_be_visible()
    page.keyboard.press("Escape")
    page.get_by_role("tab", name="Constraints", exact=True).click()
    menu = open_picker(page, "constraint")
    menu.locator(".td-opt-picker-filter").fill("efficiency_ts_with_recovery")
    expect(menu.locator('[data-opt-path="$overall_performance.efficiency_ts_with_recovery"]')).to_be_visible()
    page.keyboard.press("Escape")
    page.locator("#td-efficiency-plots").scroll_into_view_if_needed()
    page.locator('.td-theme-toggle [data-theme="dark"]').click()
    page.wait_for_function("""() => document.querySelector('#efficiency_trends_plot .js-plotly-plot')
        .data.find(trace => trace.name === 'With exhaust recovery').line.color === '#75ce9a'""")
    plasma_dark = page.locator('#loss_plot .js-plotly-plot').evaluate(
        "graph => graph.data.filter(trace => trace.meta.theme_role !== 'loss_exit_kinetic').map(trace => trace.marker.color)")
    assert plasma_dark == ['#8f0da4', '#b7318a', '#d6556d', '#ee7b51', '#fca636']
    page.screenshot(path=str(output / "diffuser_plots_dark.png"), full_page=True)
    page.locator('.td-theme-toggle [data-theme="light"]').click()
    page.wait_for_function("""() => document.querySelector('#efficiency_trends_plot .js-plotly-plot')
        .data.find(trace => trace.name === 'With exhaust recovery').line.color === '#238b62'""")
    assert page.locator('#loss_plot .js-plotly-plot').evaluate(
        "graph => graph.data.filter(trace => trace.meta.theme_role !== 'loss_exit_kinetic').map(trace => trace.marker.color)") == plasma_dark
    page.screenshot(path=str(output / "diffuser_desktop.png"), full_page=True)
    page.set_viewport_size({"width": 480, "height": 900})
    expect(page.locator("#td-metrics .td-metric")).to_have_count(5)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(output / "diffuser_mobile.png"), full_page=True)
    assert not errors, errors
    return dict(passed=True, recovery_percentages=[0, 100, 60], yaml_roundtrip_percent=75,
                headline_metrics=5, efficiency_curves=3, recovery_adjusted_loss_budget=True, errors=errors)


if __name__ == "__main__":
    harness.exercise = exercise
    harness.main()
