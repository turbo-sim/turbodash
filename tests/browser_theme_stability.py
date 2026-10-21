"""Optional browser regression for theme stability across live graph updates.

Starts its own server; uses animation-frame samples of computed paint styles,
not only the eventual Plotly layout. Pass --observe-only to record a baseline.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

from browser_picker_helpers import choose_path


ROOT = Path(__file__).resolve().parents[1]

SAMPLER = r"""() => {
    const report = {frames: 0, phases: {}, graphs: {}, failures: [], failureCount: 0};
    let active = true, phase = 'manual edit';
    const brightness = value => {
        const channels = String(value).match(/[\d.]+/g)?.map(Number);
        if (!channels || channels.length < 3 || (channels.length > 3 && channels[3] === 0)) return null;
        return (channels[0] + channels[1] + channels[2]) / (3 * 255);
    };
    const record = (element, property, role, kind) => {
        const value = getComputedStyle(element)[property], level = brightness(value);
        if (level === null) return;
        if ((kind === 'surface' && level > .65) || (kind === 'ink' && level < .5)) {
            report.failureCount++;
            if (report.failures.length < 100) report.failures.push({
                phase, frame: report.frames, role, property, value
            });
        }
    };
    const sample = () => {
        if (!active) return;
        report.frames++;
        report.phases[phase] = (report.phases[phase] || 0) + 1;
        const root = document.getElementById('td-workbench');
        if (root.dataset.theme !== 'dark') {
            report.failureCount++;
            if (report.failures.length < 100) report.failures.push({phase, frame: report.frames, role: 'root theme', value: root.dataset.theme});
        }
        record(root, 'backgroundColor', 'workbench', 'surface');
        root.querySelectorAll('.td-header,.td-inspector,.td-kpis,.td-results-disclosure,.td-plot-card,.td-section,.dash-table-container td,.dash-table-container th').forEach(element => {
            if (element.getClientRects().length) record(element, 'backgroundColor', element.id || element.className, 'surface');
        });
        root.querySelectorAll('.js-plotly-plot').forEach(graph => {
            if (!graph.getClientRects().length) return;
            const identity = graph.closest('.dash-graph')?.id || 'graph';
            report.graphs[identity] = (report.graphs[identity] || 0) + 1;
            const paper = graph.querySelector('.svg-container > svg.main-svg');
            if (paper) record(paper, 'backgroundColor', identity + ' paper', 'surface');
            graph.querySelectorAll('.bglayer > rect').forEach(element => record(element, 'fill', identity + ' plot area', 'surface'));
            graph.querySelectorAll('.xtick text,.ytick text,.g-xtitle text,.g-ytitle text,.gtitle').forEach(element => record(element, 'fill', identity + ' axis text', 'ink'));
        });
        requestAnimationFrame(sample);
    };
    window.themeStability = {
        phase(value) {phase = value;},
        stop() {active = false; return report;},
        report
    };
    requestAnimationFrame(sample);
}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromium", help="Optional installed Chromium executable")
    parser.add_argument("--output", type=Path, default=ROOT / "development/output_theme_stability")
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="Serve the committed theme assets without changing working files")
    args = parser.parse_args()
    args.output = args.output.resolve()
    for name in ("temp", "profile", "downloads", "cache"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(args.output / "temp"), TMP=str(args.output / "temp"), PYTHONDONTWRITEBYTECODE="1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    code = f"from turbodash.apps.app_turbine_optimization.app import app; app.run(port={port}, debug=False)"
    with (args.output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, "-B", "-c", code], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    urlopen(address, timeout=1).close()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Browser test server did not start")
                    time.sleep(.1)
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    str(args.output / "profile"), headless=True, viewport={"width": 1700, "height": 1100},
                    downloads_path=str(args.output / "downloads"), accept_downloads=True,
                    args=["--disable-background-networking", "--no-first-run", "--disable-component-update",
                          "--disable-crash-reporter", f"--disk-cache-dir={args.output / 'cache'}"],
                    **({"executable_path": args.chromium} if args.chromium else {}),
                )
                try:
                    report = exercise(context, address, args.output, baseline=args.baseline)
                finally:
                    context.close()
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            assert not report["browser_errors"], report["browser_errors"]
            if not args.observe_only:
                assert report["paint_samples"]["failureCount"] == 0, "Light-theme frames detected; inspect report.json"
        finally:
            server.terminate()
            server.wait(timeout=10)


def exercise(context, address, output, *, baseline=False):
    context.add_init_script("localStorage.setItem('turbodash.workbench.theme', 'dark');")
    page = context.new_page()
    intercepted_assets = []
    if baseline:
        for filename in ("workbench.js", "workbench.css"):
            body = subprocess.check_output(
                ["git", "show", f"HEAD:turbodash/apps/app_turbine_optimization/assets/{filename}"], cwd=ROOT,
            )
            content_type = "text/javascript" if filename.endswith(".js") else "text/css"
            def serve_baseline(route, *, body=body, kind=content_type, name=filename):
                intercepted_assets.append(name)
                route.fulfill(body=body, content_type=kind)
            page.route(f"**/app-assets/turbine-optimization/{filename}*", serve_baseline)
    errors, results, applied = [], [], []
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
                return
            if "result_store" in body:
                results.append(body["result_store"]["data"])
            if "opt-applied-result" in body:
                applied.append(body["opt-applied-result"]["data"])

    page.on("response", capture)
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")
    if baseline:
        assert set(intercepted_assets) == {"workbench.js", "workbench.css"}, intercepted_assets
    wait_themed(page, "dark")
    page.wait_for_timeout(200)
    assert results, "Initial calculation did not publish results"
    initial_ratio = results[-1]["inputs"]["blade_velocity_ratio"]
    page.evaluate(SAMPLER)

    def open_section(identity):
        section = page.locator(f"#{identity}")
        if section.get_attribute("open") is None:
            section.locator(":scope > summary").click()
        expect(section).to_have_js_property("open", True)

    open_section("td-section-speed")
    ratio = page.locator('input[id*=blade_velocity_ratio]')
    ratio.fill("0.61")
    ratio.blur()
    deadline = time.monotonic() + 30
    while not results or abs(results[-1]["inputs"]["blade_velocity_ratio"] - .61) > 1e-9:
        if time.monotonic() > deadline:
            raise AssertionError("Manual edit did not produce updated model results")
        page.wait_for_timeout(50)
    assert initial_ratio != .61
    wait_themed(page, "dark")
    page.wait_for_timeout(300)
    page.screenshot(path=output / "dark-after-edit.png")

    page.evaluate("themeStability.phase('optimization setup')")
    open_section("opt-panel")
    page.get_by_role("tab", name="Design variables", exact=True).click()
    choose_path(page, "variable", "blade_velocity_ratio")
    expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(1)
    numbers = page.locator('#opt-variable-rows .td-opt-row input[type="number"]')
    numbers.nth(0).fill("0.5")
    numbers.nth(2).fill("0.65")
    numbers.nth(2).blur()
    page.evaluate("themeStability.phase('optimization and apply')")
    page.locator("#opt-start").click()
    expect(page.locator("#opt-start")).to_be_disabled(timeout=10000)
    expect(page.locator("#opt-start")).to_be_enabled(timeout=60000)
    expect(page.locator("#opt-status")).to_contain_text("Optimization applied")
    expect(page.locator("#opt-log")).to_contain_text("Success: True")
    assert applied, "Optimization did not publish the applied design"
    deadline = time.monotonic() + 30
    target_ratio = applied[-1]["config"]["inputs"]["blade_velocity_ratio"]
    while abs(results[-1]["inputs"]["blade_velocity_ratio"] - target_ratio) > 1e-9:
        if time.monotonic() > deadline:
            raise AssertionError("Optimized model was not rendered")
        page.wait_for_timeout(50)
    assert abs(target_ratio - .61) > 1e-6, "Optimization must change the design"
    wait_themed(page, "dark")
    page.wait_for_timeout(400)

    page.evaluate("themeStability.phase('velocity triangle update')")
    page.locator("#td-triangle-mode").get_by_text("Velocity [m/s]", exact=True).click()
    page.wait_for_function("document.querySelector('#triangles_plot_0 .js-plotly-plot').layout.xaxis.title.text.includes('m/s')")
    wait_themed(page, "dark")
    page.wait_for_timeout(300)
    samples = page.evaluate("themeStability.stop()")
    assert samples["frames"] >= 10
    assert len(samples["graphs"]) >= 5
    page.screenshot(path=output / "dark-after-optimization.png")

    before_toggle = page.evaluate("""async () => {
        const graph = document.querySelector('#efficiency_trends_plot .js-plotly-plot');
        await Plotly.relayout(graph, {'xaxis.range': [.25, .75], 'xaxis.autorange': false});
        const snapshot = () => [...document.querySelectorAll('#meanline_pane .js-plotly-plot')].map(graph => ({
            id: graph.closest('.dash-graph').id,
            values: graph.data.map(trace => ({x: trace.x, y: trace.y})),
            colors: graph.data.map(trace => ({line: trace.line?.color, marker: trace.marker?.color, fill: trace.fillcolor}))
        }));
        window.themePlotSnapshot = snapshot;
        return snapshot();
    }""")
    page.locator('.td-theme-toggle [data-theme="light"]').click()
    wait_themed(page, "light")
    expect(page.locator("#td-workbench")).to_have_attribute("data-theme", "light")
    page.screenshot(path=output / "light-toggle.png")
    page.locator('.td-theme-toggle [data-theme="dark"]').click()
    wait_themed(page, "dark")
    after_toggle = page.evaluate("themePlotSnapshot()")
    assert after_toggle == before_toggle, "Theme round trip changed numeric data or the established trace palette"
    assert page.evaluate("document.querySelector('#efficiency_trends_plot .js-plotly-plot').layout.xaxis.range") == [.25, .75]
    return {
        "baseline_assets": baseline,
        "intercepted_assets": intercepted_assets,
        "checks": ["live input edit", "successful optimization and applied design", "velocity triangle units update",
                   "dark paint styles sampled each animation frame", "light theme still works",
                   "theme toggles preserve data, trace colors, and user zoom"],
        "initial_blade_velocity_ratio": initial_ratio,
        "optimized_blade_velocity_ratio": target_ratio,
        "published_results": len(results),
        "paint_samples": samples,
        "browser_errors": errors,
    }


def wait_themed(page, theme):
    page.wait_for_function("""theme => {
        const root = document.getElementById('td-workbench');
        const panel = getComputedStyle(root).getPropertyValue('--td-panel').trim();
        const graphs = [...root.querySelectorAll('#meanline_pane .js-plotly-plot')];
        return root.dataset.theme === theme && graphs.length >= 5 && graphs.every(graph =>
            graph._fullLayout && !graph._tdThemeBusy && graph.layout.paper_bgcolor === panel);
    }""", arg=theme, timeout=30000)


if __name__ == "__main__":
    main()
