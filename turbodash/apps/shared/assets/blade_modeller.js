/* Cumulative geometry transactions: rapid edits never depend on a server reply. */
(function () {
    "use strict";
    const edited = new WeakSet(), changedChoices = new Set();
    const numericFocus = new WeakMap();
    const drag = {active: false, path: null, row: null, epoch: null, sentAt: 0, last: null, timer: null, resolve: null};
    let journal = {epoch: null, sequence: 0, base: null, actions: []};
    function context() {
        const node = document.getElementById("blade_modeller_context");
        return {row: node && node.dataset.row, epoch: node && node.dataset.epoch};
    }
    function identifier(element) {
        if (!element || !element.id || !element.id.startsWith("{")) return null;
        try {return JSON.parse(element.id);} catch (_) {return null;}
    }
    function send(action, dragging, captured) {
        const selected = context(), source = captured || selected;
        if (!source.epoch || !source.row || source.epoch !== selected.epoch) return;
        if (journal.epoch !== source.epoch || !journal.base) return;
        action = Object.assign({}, action, {row: source.row});
        journal.sequence += 1;
        action.sequence = journal.sequence;
        journal.actions.push(action);
        const event = {epoch: journal.epoch, sequence: journal.sequence, dragging: Boolean(dragging),
                       base: journal.base, latest: action, actions: journal.actions.map(item => Object.assign({}, item))};
        window.dash_clientside.set_props("blade_modeller_events", {data: event});
        return event;
    }
    function cancel() {
        if (drag.timer !== null) clearTimeout(drag.timer);
        if (drag.resolve) drag.resolve(window.dash_clientside.no_update);
        drag.timer = null; drag.resolve = null;
    }
    function markChoice(event) {
        const field = event.target.closest(".td-blade-field");
        const id = identifier(field);
        if (id && !event.target.closest("button")) changedChoices.add(id.path);
    }
    document.addEventListener("pointerdown", function (event) {
        markChoice(event);
        const slider = event.target.closest(".td-blade-slider");
        if (!slider) return;
        edited.add(slider);
        cancel();
        Object.assign(drag, context(), {active: true, path: identifier(slider).path, sentAt: 0, last: null});
    }, true);
    const sliderKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"];
    document.addEventListener("keydown", function (event) {
        markChoice(event);
        const slider = event.target.closest(".td-blade-slider");
        if (slider && sliderKeys.includes(event.key)) edited.add(slider);
    }, true);
    // Focus-only blur must never commit a slider's step-rounded grid value.
    document.addEventListener("focusout", function (event) {
        const slider = event.target.closest(".td-blade-slider");
        if (!slider) return;
        if (!edited.has(slider)) event.stopPropagation();
        edited.delete(slider);
    }, true);
    function sliderNumber(slider) {
        const handle = slider && slider.querySelector('[role="slider"]');
        const value = handle ? Number(handle.getAttribute("aria-valuenow")) : NaN;
        return Number.isFinite(value) ? value : null;
    }
    function release() {
        if (drag.active) {
            const sliders = document.querySelectorAll(".td-blade-slider");
            const slider = Array.from(sliders).find(item => identifier(item).path === drag.path);
            const value = sliderNumber(slider);
            if (value !== null) send({kind: "set", path: drag.path, value: value}, false, drag);
        }
        drag.active = false; cancel();
    }
    document.addEventListener("pointerup", release, true);
    document.addEventListener("pointercancel", release, true);
    window.addEventListener("blur", release);
    document.addEventListener("keyup", function (event) {
        const slider = event.target.closest(".td-blade-slider");
        if (!slider || !sliderKeys.includes(event.key)) return;
        const value = sliderNumber(slider);
        if (value !== null) send({kind: "set", path: identifier(slider).path, value: value}, false);
    }, true);
    document.addEventListener("focusin", function (event) {
        const input = event.target, id = identifier(input);
        if (id && id.type === "blade-parameter" && input.matches('input[type="number"]')) {
            numericFocus.set(input, Object.assign(context(), {value: input.value}));
        }
    }, true);
    function numericCommit(event) {
        const input = event.target, id = identifier(input);
        if (!id || id.type !== "blade-parameter" || !input.matches('input[type="number"]')) return;
        const captured = numericFocus.get(input) || context();
        // An ordinary blur with no typing is a view action, not an override.
        if (captured.value === input.value) return;
        const value = input.value === "" ? null : Number(input.value);
        send({kind: "set", path: id.path, value: Number.isFinite(value) ? value : null}, false, captured);
        numericFocus.set(input, Object.assign({}, captured, {value: input.value}));
    }
    document.addEventListener("blur", numericCommit, true);
    document.addEventListener("keydown", function (event) {if (event.key === "Enter") numericCommit(event);}, true);
    function current(proposal, state, stage, row) {
        if (!proposal || !state) return false;
        const selected = state.rows.find(item => item.stage_index === (stage || 0) && item.row === (row || "stator"));
        // The journal may already contain a newer edit awaiting the server.
        if (journal.epoch === state._epoch && journal.sequence > (state.revision || 0) && proposal.token &&
            proposal.token[2] !== journal.sequence) return false;
        return selected && JSON.stringify(proposal.token) === JSON.stringify([
            state.baseline_id, state._epoch || null, state.revision || 0, selected.id
        ]);
    }
    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        bladeModeller: {
            acceptEdit: function (proposal, state, stage, row, resultTimestamp) {
                const no = window.dash_clientside.no_update, skip = Array(13).fill(no);
                if (!proposal || proposal.result_timestamp !== resultTimestamp) return skip;
                if (state && proposal.epoch === state._epoch && proposal.revision < state.revision) return skip;
                if (journal.epoch === proposal.epoch && proposal.revision < journal.sequence) return skip;
                const restoreNoUpdate = value => {
                    if (value && value._dash_no_update) return no;
                    return Array.isArray(value) ? value.map(restoreNoUpdate) : value;
                };
                const values = proposal.values.map(restoreNoUpdate);
                if (proposal.stage !== (stage || 0) || proposal.row !== (row || "stator")) {
                    // Keep edits to the previous row, but never show its controls
                    // as those of the newly selected row.
                    if (values[0] !== no) {
                        setTimeout(() => window.dash_clientside.set_props("geometry_row", {value: row || "stator"}), 0);
                    }
                    return values.slice(0, 2).concat(Array(11).fill(no));
                }
                return values;
            },
            acknowledge: function (state) {
                const no = window.dash_clientside.no_update;
                if (!state || !state._epoch) return no;
                if (journal.epoch !== state._epoch) {
                    journal = {epoch: state._epoch, sequence: state.revision || 0, base: state, actions: []};
                } else if (!journal.base || state.revision >= journal.base.revision) {
                    // Keep only operations that have not been acknowledged.
                    // Rejected attempts are acknowledged with valid geometry,
                    // so neither drag history nor errors accumulate forever.
                    journal.base = state;
                    journal.sequence = Math.max(journal.sequence, state.revision);
                    journal.actions = journal.actions.filter(item => item.sequence > state.revision);
                }
                return no;
            },
            forwardControls: function (values, resetClicks, resetRow, ids, resetIds) {
                const no = window.dash_clientside.no_update;
                const events = window.dash_clientside.callback_context.triggered || [];
                events.forEach(event => {
                    const key = event.prop_id.slice(0, event.prop_id.lastIndexOf("."));
                    let id;
                    try {id = JSON.parse(key);} catch (_) {id = key;}
                    if (id === "blade_modeller_reset" && resetRow) send({kind: "reset"}, false);
                    else if (id.type === "blade-reset") {
                        const index = resetIds.findIndex(item => item.path === id.path);
                        if (resetClicks[index]) send({kind: "reset", path: id.path}, false);
                    } else if (id.type === "blade-parameter" && changedChoices.has(id.path)) {
                        // Numeric fields have a DOM commit above. This branch
                        // handles only genuine user changes to choices/toggles.
                        if (!["camberline.interpolation", "leading_edge.pressure_side", "trailing_edge.pressure_side",
                              "correction.preserve_physical_thickness"].includes(id.path)) return;
                        changedChoices.delete(id.path);
                        const index = ids.findIndex(item => item.path === id.path);
                        send({kind: "set", path: id.path, value: values[index]}, false);
                    }
                });
                return no;
            },
            applyYaml: function (clicks, draft, captured) {
                const no = window.dash_clientside.no_update;
                if (!clicks) return no;
                const selected = context(), dialog = captured && captured.id;
                if (!captured || selected.epoch !== captured.epoch || selected.row !== captured.row) {
                    return {dialog: dialog, error: "The selected row or baseline changed. Close and reopen Row YAML."};
                }
                if (!draft || !draft.trim()) return {dialog: dialog, error: "Paste YAML or choose a file first."};
                const bytes = new TextEncoder().encode(draft);
                if (bytes.length > 1000000) return {dialog: dialog, error: "Use YAML smaller than 1 MB."};
                let binary = "";
                for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
                const event = send({kind: "import", contents: "data:application/yaml;base64," + btoa(binary)}, false, captured);
                return event ? {dialog: dialog, sequence: event.sequence} :
                    {dialog: dialog, error: "Wait for the current row to finish loading, then try again."};
            },
            sampleDrag: function (values, ids) {
                const no = window.dash_clientside.no_update;
                if (!drag.active) return no;
                const index = ids.findIndex(item => item.path === drag.path), value = values[index];
                if (value == null || value === drag.last) {cancel(); return no;}
                cancel();
                function emit() {
                    drag.sentAt = performance.now(); drag.last = value;
                    return send({kind: "set", path: drag.path, value: value}, true, drag) || no;
                }
                const delay = Math.max(0, 100 - (performance.now() - drag.sentAt));
                if (!delay) return emit();
                return new Promise(resolve => {
                    drag.resolve = resolve;
                    drag.timer = setTimeout(function () {
                        drag.timer = null; drag.resolve = null;
                        resolve(drag.active ? emit() : no);
                    }, delay);
                });
            },
            accept2d: function (proposal, state, stage, row, crosscuts) {
                const no = window.dash_clientside.no_update;
                if (!current(proposal, state, stage, row)) return [no, no, no, no, no, no];
                if (proposal.error) return [no, no, no, no, proposal.error, no];
                if (proposal.crosscuts !== (crosscuts || []).includes("show")) return [no, no, no, no, no, no];
                const figures = proposal.figures;
                return [figures.blade_to_blade, figures.meridional, figures.camberline, figures.thickness,
                        proposal.message, proposal.passage_note];
            },
            accept3d: function (proposal, state, stage, row) {
                const no = window.dash_clientside.no_update;
                if (!current(proposal, state, stage, row)) return [no, no];
                return proposal.error ? [proposal.data || no, proposal.error] : [proposal.data, proposal.message];
            }
        }
    });
}());
