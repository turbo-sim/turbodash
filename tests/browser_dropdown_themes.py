"""Audit real dropdown interactions across Meanline, Geometry and Mesh.

Run: python tests/browser_dropdown_themes.py --chromium PATH --output DIRECTORY
"""

import re

from playwright.sync_api import expect

import browser_optimization as harness
from browser_picker_helpers import choose_path, open_picker


def exercise(context, address, output):
    page = context.new_page()
    errors, checked = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(address)
    page.wait_for_selector('#perf_table .dash-table-container')
    page.locator('#opt-panel').evaluate('element => { element.open = true; }')
    page.get_by_role('tab', name='Design variables', exact=True).click()
    choose_path(page, 'variable', 'blade_velocity_ratio')
    page.get_by_role('tab', name='Constraints', exact=True).click()
    choose_path(page, 'constraint', '$overall_performance.maximum_mach_number')

    def matches_theme(element, *, input_field=False):
        return element.evaluate('''(element, inputField) => {
            const probe = document.createElement('span');
            probe.style.backgroundColor = 'var(--td-input)';
            probe.style.color = 'var(--td-ink)';
            element.parentElement.append(probe);
            const theme = getComputedStyle(probe), actual = getComputedStyle(element);
            const matches = actual.color === theme.color &&
                (inputField || actual.backgroundColor === theme.backgroundColor);
            probe.remove();
            return matches;
        }''', input_field)

    def dropdowns(theme, workspace, tab=''):
        page.locator('.td-workbench .td-opt-picker').evaluate_all(
            'elements => elements.forEach(element => { element.open = false; })')
        page.locator('.td-workbench details:not(.td-opt-picker)').evaluate_all(
            'elements => elements.forEach(element => { element.open = true; })')
        for dropdown in page.locator('.td-workbench .Select').all():
            if not dropdown.is_visible() or 'is-disabled' in (dropdown.get_attribute('class') or ''):
                continue
            identity = dropdown.evaluate('element => element.parentElement.id')
            control = dropdown.locator(':scope > .Select-control')
            control.hover(force=True)
            assert matches_theme(control), (theme, workspace, tab, identity, 'hover')
            control.click(force=True)
            assert matches_theme(control), (theme, workspace, tab, identity, 'open')
            # Select the highlighted option and leave focus in the closed control.
            page.keyboard.press('Enter')
            expect(dropdown).not_to_have_class(re.compile(r'\bis-open\b'))
            expect(dropdown).to_have_class(re.compile(r'\bis-focused\b'))
            # Moving away must not change the selected-but-focused background.
            page.mouse.move(800, 120)
            assert matches_theme(control), (theme, workspace, tab, identity, 'selected, closed and focused',
                dropdown.get_attribute('class'), control.evaluate("element => getComputedStyle(element).backgroundColor"))
            for label in dropdown.locator('.Select-value-label').all():
                assert matches_theme(label, input_field=True), (theme, identity, 'selected text')
            if 'opt-constraint-path' in identity:
                page.screenshot(path=str(output / f'constraint_selected_focused_{theme}.png'), full_page=True)
            control.click(force=True)
            # Non-searchable comparisons have a focusable DIV, not an INPUT.
            # Its inline white background was missed by outer-control checks.
            for surface in dropdown.locator('.Select-input').all():
                assert surface.evaluate("element => getComputedStyle(element).backgroundColor") == 'rgba(0, 0, 0, 0)', (
                    theme, workspace, tab, identity, 'inner focus surface')
                assert matches_theme(surface, input_field=True), (theme, workspace, tab, identity, 'inner focus text')
            if 'opt-comparison' in identity:
                page.screenshot(path=str(output / f'comparison_{theme}.png'), full_page=True)
            inputs = dropdown.locator('.Select-input input')
            if inputs.count() and inputs.first.is_editable() and inputs.first.is_visible():
                search = inputs.first
                search.fill('theme-audit-no-match')
                assert matches_theme(control), (theme, workspace, tab, identity, 'search surface')
                assert matches_theme(search, input_field=True), (theme, workspace, tab, identity, 'search text')
                search.fill('')
                search.press('Escape')
            else:
                page.keyboard.press('Escape')
            checked.append(dict(theme=theme, workspace=workspace, tab=tab, control=identity))

    for theme in ('dark', 'light'):
        page.locator(f'.td-theme-toggle [data-theme="{theme}"]').click()
        expect(page.locator('.td-workbench')).to_have_attribute('data-theme', theme)
        for tab in ('Configuration', 'Design variables', 'Constraints'):
            page.locator('.td-workspace-rail [data-workspace="meanline"]').click()
            page.get_by_role('tab', name=tab, exact=True).click()
            dropdowns(theme, 'meanline', tab)
            if tab in ('Design variables', 'Constraints'):
                kind = 'variable' if tab == 'Design variables' else 'constraint'
                for groups in (False, True):
                    menu = open_picker(page, kind, groups=groups)
                    search = menu.locator('.td-opt-picker-filter')
                    search.fill('theme-audit')
                    assert matches_theme(search), (theme, kind, groups, 'picker search')
                    search.fill('')
                    search.press('Escape')
        for workspace in ('geometry', 'mesh'):
            page.locator(f'.td-workspace-rail [data-workspace="{workspace}"]').click()
            dropdowns(theme, workspace)
        page.screenshot(path=str(output / f'dropdowns_{theme}.png'), full_page=True)
    assert any(item['workspace'] == 'geometry' for item in checked)
    assert any(item['workspace'] == 'mesh' for item in checked)
    assert any(item['tab'] == 'Design variables' for item in checked)
    assert any(item['tab'] == 'Constraints' for item in checked)
    assert not errors, errors
    return dict(passed=True, dropdown_interactions=len(checked), checked=checked, errors=errors)


if __name__ == '__main__':
    harness.exercise = exercise
    harness.main()
