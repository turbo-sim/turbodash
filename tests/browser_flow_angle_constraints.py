"""Real-browser coverage of flow-angle and minimum-turning constraints."""

import yaml
from playwright.sync_api import expect

import browser_optimization as harness
from browser_picker_helpers import choose_group, choose_path


def exercise(context, address, output):
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(address)
    page.wait_for_selector('#perf_table .dash-table-container')
    page.locator('#opt-panel').evaluate('element => { element.open = true; }')
    page.get_by_role('tab', name='Constraints', exact=True).click()
    rows = page.locator('#opt-constraint-rows .td-opt-row')
    for path in ('$stages_performance.0.flow_stations.0.alpha',
                 '$stages_performance.*.flow_stations.2.beta'):
        choose_path(page, 'constraint', path)
        expect(rows).to_have_count(1)
        choose_path(page, 'constraint', path, selected=False)
        expect(rows).to_have_count(0)
    choose_group(page, 'constraint', 'minimum_turning')
    expect(rows).to_have_count(2)
    for row in rows.all():
        expect(row.locator('.Select-value-label').last).to_have_text('≥')
        expect(row.locator('input[type="number"]').first).to_have_js_property('valueAsNumber', 50)
        expect(row.locator('input[type="number"]').last).to_have_js_property('valueAsNumber', 90)
    target = rows.first.locator('input[type="number"]').first
    target.fill('35.125')
    target.blur()
    choose_group(page, 'constraint', 'stator_inlet_relative_angle')
    choose_group(page, 'constraint', 'rotor_inlet_relative_angle')
    expect(rows).to_have_count(4)
    expect(target).to_have_js_property('valueAsNumber', 35.125)
    with page.expect_download() as download:
        page.locator('#save_button').click()
    path = output / 'flow_angle_constraints.yaml'
    download.value.save_as(path)
    saved = yaml.safe_load(path.read_text(encoding='utf-8'))
    constraints = saved['optimization']['constraints']
    assert len(constraints) == 4
    assert constraints[0] == dict(variable='$stages_performance.*.geometry_meanline.stator.flow_turning',
                                  type='>=', value=35.125, normalize=90)
    assert constraints[1]['variable'].endswith('.rotor.flow_turning')
    assert constraints[2]['variable'] == '$stages_performance.*.flow_stations.0.beta'
    assert constraints[3]['variable'] == '$stages_performance.*.flow_stations.2.beta'
    choose_group(page, 'constraint', 'minimum_turning', selected=False)
    expect(rows).to_have_count(2)
    page.locator('#load_button input[type="file"]').set_input_files({
        'name': 'flow_angle_constraints.yaml', 'mimeType': 'application/yaml',
        'buffer': path.read_bytes(),
    })
    expect(rows).to_have_count(4)
    expect(rows.first.locator('input[type="number"]').first).to_have_js_property('valueAsNumber', 35.125)
    page.locator('.td-theme-toggle [data-theme="dark"]').click()
    expect(page.locator('.td-workbench')).to_have_attribute('data-theme', 'dark')
    rows.first.scroll_into_view_if_needed()
    page.screenshot(path=str(output / 'turning_constraints_dark.png'), full_page=True)
    assert not errors, errors
    return dict(passed=True, configured_constraints=4, yaml_roundtrip=True, errors=errors)


if __name__ == '__main__':
    harness.exercise = exercise
    harness.main()
