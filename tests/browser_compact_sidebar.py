"""Check compact sidebar geometry and checklist access in a real browser."""

import argparse
import json
import os
from pathlib import Path

from browser_picker_helpers import choose_path, open_picker


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8067")
    args = parser.parse_args()
    output = ROOT / "development/app_design/output_compact_sidebar"
    scratch = output / "output_compact_sidebar"
    for name in ("temp", "profile", "downloads"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(scratch / "temp"), TMP=str(scratch / "temp"))
    from playwright.sync_api import sync_playwright, expect

    checks, errors = [], []
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(scratch / "profile"),
            executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",
            headless=True, viewport={"width": 1440, "height": 1000},
            downloads_path=str(scratch / "downloads"),
            args=["--disable-background-networking", "--no-first-run", "--disable-component-update"],
        )
        try:
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url)
            page.evaluate("localStorage.removeItem('turbodash.workbench.theme'); localStorage.removeItem('turbodash.workbench.inspector-width')")
            page.reload()
            page.wait_for_selector("#perf_table .dash-table-container", timeout=90000)
            page.locator("#opt-panel > summary").click()
            page.get_by_role("tab", name="Design variables", exact=True).click()
            paths = page.locator("#opt-variable-picker [data-opt-path]").evaluate_all("nodes => nodes.slice(0,4).map(node => node.dataset.optPath)")
            assert len(paths) == 4
            for path in paths:
                choose_path(page, "variable", path)
            expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(4)
            precision = page.locator('#opt-variable-rows .td-opt-row').first.locator('input[type="number"]').nth(1)
            precision.fill("0.6123456789123")
            precision.blur()
            expect(precision).to_have_js_property("valueAsNumber", 0.6123456789123)

            def reveal_inspector():
                if not page.locator("#design-controls").is_visible():
                    page.locator("#td-inspector-toggle").click()

            def assert_rows(kind, width):
                metrics = page.locator(f"#opt-{kind}-rows .td-opt-row").evaluate_all("""rows => rows.map(row => {
                    const bounds = row.querySelector('.td-opt-bounds');
                    const controls = Array.from(bounds.children).map(field => field.querySelector('input[type=number],.Select-control').getBoundingClientRect());
                    return {width: row.getBoundingClientRect().width, scroll: row.scrollWidth, client: row.clientWidth,
                        y: controls.map(rect => rect.y), x: controls.map(rect => rect.x), controlWidths: controls.map(rect => rect.width)};
                })""")
                assert metrics and all(max(row['y']) - min(row['y']) < 1 for row in metrics), metrics
                assert all(row['x'] == sorted(row['x']) and row['scroll'] <= row['client'] + 1 for row in metrics), metrics
                checks.append(dict(check=f"{kind} compact rows", viewport=width, rows=len(metrics),
                                   minControlWidth=min(min(row['controlWidths']) for row in metrics)))

            def menu_metrics(kind):
                menu = open_picker(page, kind)
                page.wait_for_timeout(100)
                return menu, menu.locator('.td-opt-picker-menu').evaluate("""menu => {
                    const rect = menu.getBoundingClientRect();
                    const scroll = document.querySelector('.td-inspector-scroll').getBoundingClientRect();
                    return {top: rect.top, bottom: rect.bottom, height: rect.height, inspectorTop:scroll.top,
                        inspectorBottom:scroll.bottom, clipped:rect.bottom > scroll.bottom + 1, left:rect.left, right:rect.right,
                        scrollHeight:menu.scrollHeight, clientHeight:menu.clientHeight};
                }""")

            for width, height in ((1440, 1000), (390, 844), (320, 740)):
                page.set_viewport_size(dict(width=width, height=height))
                reveal_inspector()
                for theme in ("light", "dark"):
                    page.locator(f'.td-theme-toggle [data-theme="{theme}"]').click()
                    page.get_by_role("tab", name="Design variables", exact=True).click()
                    page.locator("#opt-variable-rows").scroll_into_view_if_needed()
                    assert_rows("variable", width)
                    expect(precision).to_have_js_property("valueAsNumber", 0.6123456789123)
                    page.locator("#opt-variable-menu > summary").scroll_into_view_if_needed()
                    menu, metrics = menu_metrics("variable")
                    checks.append(dict(check="menu geometry", viewport=width, theme=theme, **metrics))
                    menu.locator(".td-opt-picker-filter").fill(paths[0])
                    expect(menu.locator(f'[data-opt-path="{paths[0]}"]')).to_be_visible()
                    page.screenshot(path=scratch / f"workbench_sidebar_{width}_{theme}.png", full_page=True)
                    menu.locator(".td-opt-picker-filter").press("Escape")
                    if width == 1440 and theme == "light":
                        page.screenshot(path=output / "workbench_optimization.png", full_page=True)
                    page.get_by_role("tab", name="Constraints", exact=True).click()
                    constraints = page.locator("#opt-constraint-picker [data-opt-path]").evaluate_all("nodes => nodes.slice(0,3).map(node => node.dataset.optPath)")
                    for path in constraints:
                        choose_path(page, "constraint", path)
                    assert_rows("constraint", width)
                    if width == 1440 and theme == "light":
                        page.screenshot(path=output / "workbench_optimization_constraints.png", full_page=True)
                    label = page.locator("#opt-constraint-rows .td-opt-field-label[title]").first
                    label.focus()
                    assert label.evaluate("node => getComputedStyle(node, '::after').display") == "block"
                    page.locator("#opt-constraint-menu > summary").scroll_into_view_if_needed()
                    menu, metrics = menu_metrics("constraint")
                    checks.append(dict(check="constraint menu geometry", viewport=width, theme=theme, **metrics))
                    menu.locator(".td-opt-picker-filter").press("Escape")
                    assert page.locator(".td-inspector-scroll").evaluate("node => node.scrollWidth <= node.clientWidth + 1")
                    expect(page.locator("#opt-start")).to_be_in_viewport()
            page.set_viewport_size(dict(width=1440, height=1000))
            page.locator('.td-theme-toggle [data-theme="light"]').click()
            page.get_by_role("tab", name="Design variables", exact=True).click()
            page.locator("#opt-variable-group-menu > summary").scroll_into_view_if_needed()
            groups = open_picker(page, "variable", groups=True)
            groups.locator('.td-opt-picker-branch').evaluate_all("nodes => nodes.forEach(node => node.open = true)")
            page.screenshot(path=output / "workbench_optimization_groups.png", full_page=True)
            groups.locator('.td-opt-picker-filter').press("Escape")
            page.locator('.td-workbench').evaluate("node => node.style.setProperty('--td-inspector-width', '280px')")
            assert_rows("variable", "desktop inspector 280")
            page.locator('.td-workbench').evaluate("node => node.style.removeProperty('--td-inspector-width')")

            # Create a long inspector so the selector can sit just above its dock.
            page.locator('.td-inspector .td-section:not(#opt-panel)').evaluate_all("nodes => nodes.forEach(node => node.open = true)")
            for width, height in ((1440, 1000), (390, 844), (320, 740)):
                page.set_viewport_size(dict(width=width, height=height))
                reveal_inspector()
                page.locator('.td-inspector-scroll').evaluate("scroll => scroll.scrollTop = 0")
                fields = page.locator('.td-field').evaluate_all("""nodes => nodes.filter(node => node.getBoundingClientRect().height).map(node => {
                    const label = node.querySelector('.td-field-label') || node.querySelector('label');
                    const control = node.querySelector('.td-field-control') || node.querySelector('.td-stepper');
                    const a = label.getBoundingClientRect(), b = control.getBoundingClientRect();
                    return {inline: a.right <= b.left + 1, centered:Math.abs(a.y+a.height/2-b.y-b.height/2)<1,
                        contained:node.scrollWidth <= node.clientWidth+1};
                })""")
                assert fields and all(field['inline'] and field['centered'] and field['contained'] for field in fields), fields
                cards = page.locator('.td-inspector .td-section[open]').evaluate_all("""nodes => nodes.map(node => {
                    const css = getComputedStyle(node), body = node.querySelector('.td-section-body,.td-opt-body');
                    const a=node.getBoundingClientRect(), b=body.getBoundingClientRect();
                    return css.borderLeftWidth === '1px' && css.borderLeftColor !== 'rgba(0, 0, 0, 0)' && b.left >= a.left && b.right <= a.right;
                })""")
                assert cards and all(cards), cards
                checks.append(dict(check="inline fields and expanded cards", viewport=width, fields=len(fields), cards=len(cards)))
                page.screenshot(path=scratch / f"workbench_sidebar_fields_{width}.png", full_page=True)
                page.locator('.td-inspector-scroll').evaluate("""scroll => {
                    const target=document.querySelector('#opt-variable-menu > summary');
                    scroll.scrollTop += target.getBoundingClientRect().top - (scroll.getBoundingClientRect().bottom - 45);
                }""")
                menu, metrics = menu_metrics("variable")
                checks.append(dict(check="near-bottom menu", viewport=width, **metrics))
                page.screenshot(path=scratch / f"workbench_sidebar_bottom_{width}.png", full_page=True)
                assert not metrics['clipped'], metrics
                menu.locator('.td-opt-picker-filter').fill("")
                menu.locator('.td-opt-picker-branch').evaluate_all("nodes => nodes.forEach(node => node.open = true)")
                page.wait_for_timeout(100)
                expanded = menu.locator('.td-opt-picker-menu').evaluate("""node => {
                    const rect=node.getBoundingClientRect(), host=document.querySelector('.td-inspector-scroll').getBoundingClientRect();
                    const sticky=document.querySelector('#opt-panel > summary').getBoundingClientRect();
                    return {top:rect.top,bottom:rect.bottom,hostBottom:host.bottom,stickyBottom:sticky.bottom};
                }""")
                assert expanded['bottom'] <= expanded['hostBottom'] + 1 and expanded['top'] >= expanded['stickyBottom'], expanded
                menu.locator('.td-opt-picker-filter').press("Escape")
            assert not errors, errors
            (output / "compact_sidebar_browser_checks.json").write_text(json.dumps(dict(checks=checks, errors=errors), indent=2), encoding="utf-8")
            print(json.dumps(dict(checks=len(checks), errors=errors, clippedMenus=[check for check in checks if check.get('clipped')]), indent=2))
        finally:
            context.close()


if __name__ == "__main__":
    main()
