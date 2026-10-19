"""Real-browser optimization editor stress regression.

Uses the isolated server/browser lifecycle from browser_optimization.py.
Run with PYTHONPATH pointing to Playwright and --chromium for installed Chrome.
"""

import json

import yaml

from playwright.sync_api import expect

import browser_optimization


def exercise(context, address, output):
    page = context.new_page()
    errors, requests, responses, checks = [], [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def response_received(response):
        if "_dash-update-component" not in response.url:
            return
        body = response.request.post_data_json
        if "opt-variable-rows" in body.get("output", ""):
            requests.append(body)
            if response.status == 200:
                responses.append(response.json())
        if response.status >= 400:
            errors.append(f"Callback HTTP {response.status}: {body.get('output')}")

    page.on("response", response_received)
    try:
        page.goto(address)
        page.wait_for_selector("#perf_table .dash-table-container", timeout=60000)
        page.locator("#opt-panel > summary").click()
        page.get_by_role("tab", name="Design variables", exact=True).click()
        variables = page.locator("#opt-variable-rows .td-opt-row")
        for _ in range(4):
            page.locator("#opt-add-variable").click()
            page.wait_for_timeout(250)
        expect(variables).to_have_count(4)
        checks.append("Sequential add variable")
        network = context.new_cdp_session(page)
        network.send("Network.enable")
        network.send("Network.emulateNetworkConditions", {
            "offline": False, "latency": 350, "downloadThroughput": -1, "uploadThroughput": -1,
        })
        for _ in range(5):
            page.locator("#opt-add-variable").click()
        expect(variables).to_have_count(9)
        checks.append("Five rapid variable clicks remain five additions")
        page.get_by_role("tab", name="Constraints", exact=True).click()
        constraints = page.locator("#opt-constraint-rows .td-opt-row")
        for _ in range(5):
            page.locator("#opt-add-constraint").click()
        expect(constraints).to_have_count(5)
        checks.append("Five rapid constraint clicks remain five additions")

        # Each removal keeps its row identity even when preceding rows vanish.
        remove = constraints.get_by_role("button", name="Remove constraint").evaluate_all(
            "buttons => buttons.slice(0, 4).map(button => button.id)"
        )
        for identity in remove:
            page.locator(f"[id={json.dumps(identity)}]").click()
        expect(constraints).to_have_count(1)
        checks.append("Rapid removal of four distinct constraints")
        page.get_by_role("tab", name="Design variables", exact=True).click()
        remove = variables.get_by_role("button", name="Remove variable").evaluate_all(
            "buttons => buttons.slice(1, 8).map(button => button.id)"
        )
        for identity in remove:
            page.locator(f"[id={json.dumps(identity)}]").click()
        expect(variables).to_have_count(2)
        checks.append("Rapid removal of seven distinct variables")

        first = variables.first.locator('input[type="number"]')
        for box, value in zip([first.nth(i) for i in range(3)], ["0.52", "0.6123456", "0.65"]):
            box.fill(value)
        page.locator("#opt-add-variable").click()
        expect(variables).to_have_count(3)
        for index, value in enumerate([.52, .6123456, .65]):
            expect(first.nth(index)).to_have_js_property("valueAsNumber", value)
        checks.append("Adding a row commits and preserves unblurred precision edits")

        # Commit a new bound after an Add request already captured the old
        # State. The late response must not restore that superseded number.
        with page.expect_request(lambda request: "_dash-update-component" in request.url
                                 and "opt-variable-rows" in (request.post_data or "")):
            page.locator("#opt-add-variable").click()
        first.nth(0).fill("0.53216789")
        first.nth(0).blur()
        expect(variables).to_have_count(4)
        expect(first.nth(0)).to_have_js_property("valueAsNumber", .53216789)
        variables.last.get_by_role("button", name="Remove variable").click()
        expect(variables).to_have_count(3)
        checks.append("A late Add response preserves bounds committed after its request")

        # Change a selection while the previous request is still in flight.
        selected = variables.nth(1).locator(".Select-input input")
        selected.fill("Reaction")
        selected.press("Enter")
        expect(variables.nth(1)).to_contain_text("Reaction")
        numbers = variables.nth(1).locator('input[type="number"]')
        expect(numbers.nth(0)).to_have_js_property("valueAsNumber", 0)
        expect(numbers.nth(2)).to_have_js_property("valueAsNumber", .75)
        expect(numbers.nth(1)).to_have_value("")
        expect(first.nth(1)).to_have_js_property("valueAsNumber", .6123456)
        checks.append("Changing one variable resets its defaults and retains other rows")

        selected.fill("Stator exit flow angle")
        with page.expect_request(lambda request: "_dash-update-component" in request.url
                                 and "opt-variable-rows" in (request.post_data or "")):
            selected.press("Enter")
        numbers.nth(0).fill("55.1234567")
        numbers.nth(0).blur()
        expect(variables.nth(1)).to_contain_text("Stator exit flow angle")
        expect(numbers.nth(2)).to_have_js_property("valueAsNumber", 80)
        expect(numbers.nth(0)).to_have_js_property("valueAsNumber", 55.1234567)
        checks.append("A bound edited after path selection survives its delayed defaults response")

        page.locator("#td-section-speed > summary").click()
        ratio = page.locator('input[id*=blade_velocity_ratio]')
        ratio.fill("0.61")
        ratio.blur()
        first.nth(0).fill("0.5456789")
        expect(first.nth(1)).to_have_attribute("placeholder", "Current: 0.61", timeout=30000)
        expect(first.nth(0)).to_have_js_property("valueAsNumber", .5456789)
        expect(first.nth(0)).to_be_focused()
        first.nth(0).blur()
        checks.append("Catalog refresh preserves the focused uncommitted numeric edit")

        # Add both kinds in one event batch; no trigger can take precedence.
        page.evaluate("""() => {
            document.getElementById('opt-add-variable').click();
            document.getElementById('opt-add-constraint').click();
        }""")
        expect(variables).to_have_count(4)
        expect(constraints).to_have_count(2)
        expect(first.nth(0)).to_have_js_property("valueAsNumber", .5456789)
        checks.append("Simultaneous variable and constraint additions retain both actions")

        # In the same batch, delete from both lists and request another add.
        page.evaluate("""() => {
            document.querySelector('#opt-variable-rows .td-opt-row:last-child button').click();
            document.querySelector('#opt-constraint-rows .td-opt-row:last-child button').click();
            document.getElementById('opt-add-variable').click();
        }""")
        expect(variables).to_have_count(4)
        expect(constraints).to_have_count(1)
        expect(first.nth(0)).to_have_js_property("valueAsNumber", .5456789)
        checks.append("Simultaneous additions and removals preserve live numeric bounds")

        # All clicks are delivered to the same fixed screen position.
        location = page.locator("#opt-add-variable").bounding_box()
        page.locator("#opt-add-variable").click(click_count=3, delay=75)
        expect(variables).to_have_count(7)
        assert page.locator("#opt-add-variable").bounding_box() == location
        checks.append("Add control remains stationary as rows grow")

        network.send("Network.emulateNetworkConditions", {
            "offline": False, "latency": 0, "downloadThroughput": -1, "uploadThroughput": -1,
        })
        page.get_by_role("tab", name="Constraints", exact=True).click()
        fields = constraints.first.locator('input[type="number"]')
        for scale, diagnostic in [("", "Constraint scale must be a finite number"),
                                  ("0", "Constraint scale must be positive")]:
            fields.nth(1).fill(scale)
            page.locator("#opt-start").click()
            expect(page.locator("#opt-status")).to_contain_text(diagnostic)
            expect(page.locator("#opt-start")).to_be_enabled()
        fields.nth(1).fill("10")
        fields.nth(0).fill("")
        page.locator("#opt-start").click()
        expect(page.locator("#opt-status")).to_contain_text("Target must be a finite number")
        expect(page.locator("#opt-start")).to_be_enabled()
        checks.append("Blank target, blank scale, and zero scale give recoverable validation errors")

        constraints.get_by_role("button", name="Remove constraint").click()
        expect(constraints).to_have_count(0)
        page.get_by_role("tab", name="Design variables", exact=True).click()
        while variables.count():
            count = variables.count()
            variables.last.get_by_role("button", name="Remove variable").click()
            expect(variables).to_have_count(count - 1)
        page.locator("#opt-start").click()
        expect(page.locator("#opt-status")).to_contain_text("Add at least one design variable")
        page.locator("#opt-add-variable").click()
        expect(variables).to_have_count(1)
        fields = variables.first.locator('input[type="number"]')
        fields.nth(0).fill("0.5")
        fields.nth(2).fill("0.65")
        page.locator("#opt-start").click()
        expect(page.locator("#opt-start")).to_be_disabled()
        expect(page.locator("#opt-status")).to_contain_text("Optimization applied", timeout=60000)
        expect(page.locator("#opt-start")).to_be_enabled()
        checks.append("Removing all rows and correcting validation errors permits a successful fresh run")

        def open_variables():
            section = page.locator("#opt-panel")
            if section.get_attribute("open") is None:
                section.locator(":scope > summary").click()
            page.get_by_role("tab", name="Design variables", exact=True).click()

        config = yaml.safe_load((browser_optimization.ROOT / "development/optimization/case_axial.yaml").read_text(encoding="utf-8"))
        config["optimization"]["design_variables"] = {
            "blade_velocity_ratio": dict(lower_value=.5, upper_value=.75, initial_value=.6),
        }
        config["optimization"]["constraints"] = []
        page.evaluate("""() => {
            const fetch = window.fetch.bind(window);
            window.tdEditorLoadCount = 0;
            window.tdEditorLoadDelay = 0;
            window.fetch = (...args) => {
                let delay = 0;
                const body = args[1]?.body;
                if (typeof body === 'string' && body.includes('opt-editor-state.data')) {
                    const payload = JSON.parse(body);
                    if (payload.changedPropIds.includes('opt-editor-load.data')) {
                        window.tdEditorLoadCount += 1;
                        delay = window.tdEditorLoadDelay;
                    }
                }
                return fetch(...args).then(response => delay ?
                    new Promise(resolve => setTimeout(() => resolve(response), delay)) : response);
            };
        }""")

        def upload(filename):
            before = page.evaluate("window.tdEditorLoadCount")
            previous_design = page.locator("#td-workbench").get_attribute("data-design-revision")
            page.locator('#load_button input[type="file"]').set_input_files({
                "name": filename, "mimeType": "text/yaml",
                "buffer": (yaml.safe_dump(config) + "# " + filename + "\n").encode(),
            })
            page.wait_for_function("before => window.tdEditorLoadCount > before", arg=before)
            page.wait_for_function("before => document.getElementById('td-workbench').dataset.designRevision !== before",
                                   arg=previous_design)
            open_variables()

        open_variables()
        old_identity = variables.first.locator('input[type="number"]').first.get_attribute("id")
        upload("baseline-editor-load.yaml")
        expect(variables).to_have_count(1)
        expect(variables.first.locator('input[type="number"]').first).not_to_have_attribute("id", old_identity)
        expect(variables.first.locator('input[type="number"]').first).to_have_js_property("valueAsNumber", .5)
        page.evaluate("window.tdEditorLoadDelay = 750")
        config["optimization"]["design_variables"]["blade_velocity_ratio"]["lower_value"] = .57
        upload("changed-bounds-same-turbine.yaml")
        page.locator("#opt-add-variable").click()
        expect(variables).to_have_count(2)
        expect(variables.first.locator('input[type="number"]').first).to_have_js_property("valueAsNumber", .57)
        checks.append("Repeated loading with unchanged turbine inputs applies new bounds before pending Add")

        lower = variables.first.locator('input[type="number"]').first
        with page.expect_response(lambda response: "_dash-update-component" in response.url
                                  and "opt-editor-state.data" in (response.request.post_data or "")):
            lower.fill("0.53")
            lower.blur()
        upload("reset-identical-design.yaml")
        page.locator("#opt-add-variable").click()
        expect(variables).to_have_count(2)
        expect(variables.first.locator('input[type="number"]').first).to_have_js_property("valueAsNumber", .57)
        checks.append("Reloading identical settings resets manual bounds and retains a later Add")
        assert not errors, errors
        return {"checks": checks, "errors": errors}
    finally:
        (output / "editor_requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
        (output / "editor_responses.json").write_text(json.dumps(responses, indent=2), encoding="utf-8")
        page.screenshot(path=output / "editor_stress.png")


if __name__ == "__main__":
    browser_optimization.exercise = exercise
    browser_optimization.main()
