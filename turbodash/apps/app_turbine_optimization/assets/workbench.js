/* View-only interactions. Numeric model controls are never edited here. */
(function () {
    "use strict";
    const THEME_KEY = "turbodash.workbench.theme";
    const WIDTH_KEY = "turbodash.workbench.inspector-width";
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    let root = null;
    let selectedResult = 0;
    let matches = [];
    let pendingRefresh = false;
    let pendingThemeRefresh = false;
    let explicitTheme = readPreference(THEME_KEY);
    let currentTheme = explicitTheme || (media.matches ? "dark" : "light");
    const numericInputs = new WeakMap();
    const typedNumbers = new Map();

    function paddedNumber(text) {
        // Work on the entered digits: numeric rounding/toFixed would discard
        // significant digits, deliberate trailing zeros, and tiny tolerances.
        const match = /^(-?)(\d*)(?:\.(\d*))?([eE][+-]?\d+)?$/.exec(text);
        if (!match || !(match[2] || match[3]) || !Number.isFinite(Number(text))) return text;
        const fraction = match[3] || "";
        return match[1] + (match[2] || "0") + "." + fraction.padEnd(3, "0") + (match[4] || "");
    }
    function continuousNumber(input) {
        return input instanceof HTMLInputElement && input.type === "number" && input.step === "any";
    }
    function scientificNumber(text) {
        // Move the decimal in the written digits rather than rounding the value.
        const match = /^(-?)(\d*)(?:\.(\d*))?([eE][+-]?\d+)?$/.exec(text);
        if (!match || !Number.isFinite(Number(text))) return text;
        const digits = (match[2] || "") + (match[3] || "");
        const first = digits.search(/[1-9]/);
        if (first < 0) return "0.000e+0";
        const exponent = Number((match[4] || "e0").slice(1)) + match[2].length - first - 1;
        return match[1] + digits[first] + "." + digits.slice(first + 1).padEnd(3, "0") +
            "e" + (exponent >= 0 ? "+" : "") + exponent;
    }
    function formatNumericInput(input) {
        if (!root || !root.contains(input) || !continuousNumber(input) || document.activeElement === input ||
            input.value === "" || !input.validity.valid) return;
        const record = numericInputs.get(input);
        if (!record) return;
        const raw = input.value, typed = input.id && typedNumbers.get(input.id);
        // Numeric callbacks naturally drop lexical trailing zeros. Retain the
        // user's spelling while it still denotes the same committed value.
        const spelling = typed && Number(typed) === Number(raw) ? typed : raw;
        const display = input.classList.contains("td-scientific") ? scientificNumber(spelling) : paddedNumber(spelling);
        if (display !== raw) record.write.call(input, display);
    }
    function scheduleNumericFormat(input) {
        const record = numericInputs.get(input);
        if (!record || record.pending) return;
        record.pending = true;
        queueMicrotask(() => { record.pending = false; formatNumericInput(input); });
    }
    function formatNumericInputs() {
        if (!root) return;
        root.querySelectorAll('input[type="number"][step="any"]').forEach(input => {
            if (!numericInputs.has(input)) {
                // DCC and Bootstrap number inputs write their native value
                // property directly. Preserve their existing setter (including
                // React's tracking) and observe only this mounted input; no
                // prototype patch, polling, synthetic events, or Dash setProps.
                const descriptor = Object.getOwnPropertyDescriptor(input, "value") ||
                    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
                if (!descriptor || !descriptor.get || !descriptor.set) return;
                numericInputs.set(input, { write: descriptor.set, pending: false });
                if (descriptor.configurable !== false) {
                    Object.defineProperty(input, "value", {
                        configurable: true, enumerable: descriptor.enumerable,
                        get() { return descriptor.get.call(this); },
                        set(value) { descriptor.set.call(this, value); scheduleNumericFormat(this); }
                    });
                }
            }
            formatNumericInput(input);
        });
    }

    function readPreference(key) {
        try { return window.localStorage.getItem(key); } catch (_) { return null; }
    }
    function savePreference(key, value) {
        try { window.localStorage.setItem(key, value); } catch (_) { /* Private browser: session preference still works. */ }
    }
    function setProps(id, properties) {
        if (window.dash_clientside && typeof window.dash_clientside.set_props === "function") {
            window.dash_clientside.set_props(id, properties);
        }
    }
    function normalized(value) {
        return String(value || "").normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
            .replace(/α/g, " alpha ").replace(/β/g, " beta ").replace(/η/g, " efficiency ")
            .replace(/ω/g, " omega speed ").replace(/ν/g, " nu velocity ").replace(/ρ/g, " rho density ")
            .replace(/_/g, " ").toLowerCase().replace(/\s+/g, " ").trim();
    }
    function currentStyle(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
    function labelDropdowns() {
        if (!root) return;
        root.querySelectorAll("[data-opt-help-tab]").forEach(description => {
            const tab = root.querySelector("." + description.dataset.optHelpTab);
            if (tab) {
                tab.dataset.tooltip = description.textContent;
                tab.setAttribute("aria-describedby", description.id);
            }
        });
        root.querySelectorAll(".Select input").forEach(input => {
            const field = input.closest(".td-field,.td-parameter,.td-opt-field");
            const label = field && (field.matches("label") ? field.querySelector("span") : field.querySelector("label"));
            if (label) input.setAttribute("aria-label", label.textContent.trim());
            else if (input.closest("#opt-variable-rows")) input.setAttribute("aria-label", "Design variable");
            else if (input.closest("#opt-constraint-rows")) input.setAttribute("aria-label", "Constraint quantity");
        });
    }
    function applyTheme(theme, persist) {
        if (!root) return;
        currentTheme = theme === "dark" ? "dark" : "light";
        root.dataset.theme = currentTheme;
        root.querySelectorAll(".td-theme-toggle [data-theme]").forEach(button => {
            button.setAttribute("aria-pressed", String(button.dataset.theme === currentTheme));
        });
        if (persist) { explicitTheme = currentTheme; savePreference(THEME_KEY, currentTheme); }
        themeFigures();
    }
    function themeFigure(graph) {
        if (!window.Plotly || !graph._fullLayout || graph._tdThemeBusy) return;
        const panel = currentStyle("--td-panel"), ink = currentStyle("--td-ink");
        const grid = currentStyle("--td-grid"), line = currentStyle("--td-line");
        // A new server figure may replace layout while keeping the same DOM node.
        if (graph.layout && graph.layout.paper_bgcolor === panel && graph.layout.font &&
            graph.layout.font.color === ink && graph._tdTheme === currentTheme) return;
        const update = {
            paper_bgcolor: panel, plot_bgcolor: panel, "font.color": ink,
            "font.family": "Segoe UI, sans-serif", "legend.bgcolor": panel,
            "legend.font.color": ink, "hoverlabel.bgcolor": panel,
            "hoverlabel.font.color": ink, "hoverlabel.bordercolor": line
        };
        Object.keys(graph._fullLayout).filter(key => /^[xy]axis\d*$/.test(key)).forEach(axis => {
            update[axis + ".color"] = ink;
            update[axis + ".linecolor"] = line;
            update[axis + ".gridcolor"] = grid;
            update[axis + ".zerolinecolor"] = line;
            update[axis + ".title.font.color"] = ink;
            update[axis + ".tickfont.color"] = ink;
        });
        Object.keys(graph._fullLayout).filter(key => /^scene\d*$/.test(key)).forEach(scene => {
            update[scene + ".bgcolor"] = panel;
            ["xaxis", "yaxis", "zaxis"].forEach(axis => {
                update[scene + "." + axis + ".color"] = ink;
                update[scene + "." + axis + ".gridcolor"] = grid;
                update[scene + "." + axis + ".backgroundcolor"] = panel;
            });
        });
        (graph.layout.annotations || []).forEach((annotation, index) => {
            update["annotations[" + index + "].font.color"] = ink;
            if (annotation.showarrow) update["annotations[" + index + "].arrowcolor"] = ink;
        });
        const neutral = /^(black|#000(?:000)?|#18314d|#e7eff9|#647892|#a0b2ca|#23364d|#e2e8f1)$/i;
        (graph.layout.shapes || []).forEach((shape, index) => {
            if (shape.line && neutral.test(shape.line.color)) update["shapes[" + index + "].line.color"] = ink;
        });
        const traceUpdates = {};
        const lossColors = {
            loss_profile: ["#2462D1", "#88B7FF"],
            loss_trailing: ["#4B84E5", "#6D9DE8"],
            loss_secondary: ["#7AA6EC", "#B3CEFF"],
            loss_clearance: ["#FF7F0E", "#FFAD5C"],
            loss_incidence: ["#F5AD66", "#F3C48F"],
            loss_exit_kinetic: ["#647892", "#A0B2CA"],
            loss_unallocated: ["#8995A5", "#BCC6D4"]
        };
        (graph.data || []).forEach((trace, index) => {
            const role = trace.meta && trace.meta.theme_role, colors = {};
            if (role === "endwall") colors["line.color"] = currentTheme === "dark" ? "#a0b2ca" : "#647892";
            else if (trace.line && neutral.test(trace.line.color)) colors["line.color"] = ink;
            else if (trace.line && ["#1f77b4", "#69a9e2"].includes(trace.line.color)) colors["line.color"] = currentTheme === "dark" ? "#69a9e2" : "#1f77b4";
            else if (trace.line && ["#ff7f0e", "#ffad5c"].includes(trace.line.color)) colors["line.color"] = currentTheme === "dark" ? "#ffad5c" : "#ff7f0e";
            if (trace.marker && typeof trace.marker.color === "string" && neutral.test(trace.marker.color)) colors["marker.color"] = ink;
            else if (trace.marker && ["#1f77b4", "#69a9e2"].includes(trace.marker.color)) colors["marker.color"] = currentTheme === "dark" ? "#69a9e2" : "#1f77b4";
            else if (trace.marker && ["#ff7f0e", "#ffad5c"].includes(trace.marker.color)) colors["marker.color"] = currentTheme === "dark" ? "#ffad5c" : "#ff7f0e";
            if (lossColors[role]) {
                colors["marker.color"] = lossColors[role][currentTheme === "dark" ? 1 : 0];
                colors["marker.line.color"] = panel;
            }
            if (role === "solid") colors.fillcolor = currentTheme === "dark" ? "rgba(160,178,202,0.12)" : "rgba(100,120,146,0.12)";
            if (role === "channel") colors.fillcolor = currentTheme === "dark" ? "rgba(136,183,255,0.08)" : "rgba(36,98,209,0.05)";
            if (role === "flow_guide") colors["line.color"] = currentTheme === "dark" ? "rgba(128,183,255,0.58)" : "rgba(68,118,188,0.48)";
            Object.entries(colors).forEach(([key, color]) => {
                // Undefined skips untouched traces, including numeric colour maps.
                if (!traceUpdates[key]) traceUpdates[key] = new Array(graph.data.length).fill(undefined);
                traceUpdates[key][index] = color;
            });
        });
        graph._tdThemeBusy = true;
        graph._tdTheme = currentTheme;
        // Update traces and framing together instead of redrawing once per colour.
        Promise.resolve().then(() => window.Plotly.update(graph, traceUpdates, update))
            .finally(() => { graph._tdThemeBusy = false; queueThemeRefresh(); });
    }
    function queueThemeRefresh() {
        if (pendingThemeRefresh) return;
        pendingThemeRefresh = true;
        queueMicrotask(() => { pendingThemeRefresh = false; themeFigures(); });
    }
    function themeFigures() {
        if (!root) return;
        root.querySelectorAll(".js-plotly-plot").forEach(graph => {
            if (!graph._tdThemeListener && typeof graph.on === "function") {
                graph.on("plotly_afterplot", queueThemeRefresh);
                graph._tdThemeListener = true;
            }
            if (!graph._tdRadialResetListener && typeof graph.on === "function" &&
                graph.closest("#meridional_plot,#geometry_meridional,#mesh_meridional")) {
                // A Dash graph mounts before its first figure. Plotly has no initial
                // range snapshot in that case, and its fixed-aspect reset can shift
                // an autoranged radial baseline. Correct only autoscale/reset;
                // explicit user zoom ranges remain entirely under user control.
                graph.on("plotly_relayout", event => {
                    if (event["yaxis.autorange"] !== true || graph._tdResettingRadial) return;
                    requestAnimationFrame(() => {
                        const range = graph._fullLayout.yaxis.range;
                        if (!range || range[0] === 0 || !Number.isFinite(range[1]) || range[1] <= 0) return;
                        graph._tdResettingRadial = true;
                        Promise.resolve(window.Plotly.relayout(graph, {"yaxis.range": [0, range[1]], "yaxis.autorange": false}))
                            .finally(() => { graph._tdResettingRadial = false; });
                    });
                });
                graph._tdRadialResetListener = true;
            }
            themeFigure(graph);
        });
    }
    function setupHeader() {
        const header = root.querySelector(".td-header");
        const measure = () => {
            root.style.setProperty("--td-inspector-top", header.getBoundingClientRect().bottom + "px");
        };
        const observer = new ResizeObserver(measure);
        observer.observe(header);
        measure();
    }
    function setWorkspace(workspace) {
        if (!root || !["meanline", "geometry", "mesh"].includes(workspace)) return;
        root.dataset.workspace = workspace;
        root.querySelectorAll(".td-workspace-rail [data-workspace]").forEach(button => {
            const active = button.dataset.workspace === workspace;
            button.classList.toggle("active", active);
            button.setAttribute("aria-current", active ? "page" : "false");
        });
        setProps("td-workspace", { data: workspace });
        applyScope();
        window.setTimeout(() => {
            if (window.Plotly) root.querySelectorAll(".js-plotly-plot").forEach(graph => {
                if (graph.offsetParent && graph._fullLayout) window.Plotly.Plots.resize(graph);
            });
        }, 60);
    }
    function applyScope() {
        if (!root) return;
        const stage = String(root.dataset.stageIndex || "0");
        const row = root.dataset.bladeRow || "stator";
        root.querySelectorAll(".td-inspector [data-stage-index]").forEach(element => {
            element.hidden = String(element.dataset.stageIndex) !== stage;
        });
        root.querySelectorAll(".td-inspector [data-blade-row]").forEach(element => {
            element.hidden = element.dataset.bladeRow !== row;
        });
        root.querySelectorAll("[data-controls-workspace]").forEach(element => {
            element.hidden = element.dataset.controlsWorkspace !== (root.dataset.workspace || "meanline");
        });
    }
    function setDrawer(open) {
        root.classList.toggle("td-inspector-open", open);
        const button = document.getElementById("td-inspector-toggle");
        if (button) button.setAttribute("aria-expanded", String(open));
    }
    function identityFor(element) {
        let control = element.matches("input,select,textarea,[role='combobox']") ? element :
            element.querySelector("input:not([type=hidden]),select,textarea,[role='combobox']");
        if (!control && element.dataset.parameterLabel) control = element.querySelector("button");
        const identified = element.id ? element : control;
        let identity = {};
        try { identity = JSON.parse((identified && identified.id) || "{}"); } catch (_) { /* Ordinary string ID. */ }
        if (!identity.scope && control) {
            try { identity = JSON.parse(control.id || "{}"); } catch (_) { /* Ordinary string ID. */ }
        }
        return { control, identity };
    }
    function indexFields() {
        const index = [], seen = new Set();
        const candidates = root.querySelectorAll("[data-parameter-label],.td-field,.td-parameter,.td-opt-field,input[type=number],select,.dash-dropdown");
        candidates.forEach(element => {
            if (element.closest("#td-search-results") || element.id === "td-stage-selector") return;
            const wrapper = element.closest("[data-parameter-label],.td-field,.td-parameter,.td-opt-field") || element;
            if (seen.has(wrapper)) return;
            seen.add(wrapper);
            const { control, identity } = identityFor(wrapper);
            if (!control && !wrapper.querySelector(".Select")) return;
            const labelElement = wrapper.matches("label") ? wrapper.querySelector("span") : wrapper.querySelector("label");
            let label = labelElement ? labelElement.textContent : wrapper.dataset.parameterLabel;
            if (!label) {
                if (control && control.labels && control.labels.length) label = control.labels[0].textContent;
                else label = wrapper.getAttribute("aria-label") || (control && control.getAttribute("aria-label"));
            }
            if (!label && identity.key) label = identity.key.replace(/[_.]/g, " ");
            if (!label) return;
            let stageElement = wrapper.closest("[data-stage-index]");
            if (stageElement === root) stageElement = null;
            let rowElement = wrapper.closest("[data-blade-row]");
            if (rowElement === root) rowElement = null;
            const stage = stageElement ? Number(stageElement.dataset.stageIndex) : identity.stage;
            const row = rowElement ? rowElement.dataset.bladeRow : identity.row ||
                (identity.key && /^(stator|rotor)\./.test(identity.key) ? identity.key.split(".")[0] : null);
            const controlsWorkspace = wrapper.closest("[data-controls-workspace]");
            const pane = wrapper.closest("#geometry_pane,#mesh_pane");
            const workspace = controlsWorkspace ? controlsWorkspace.dataset.controlsWorkspace :
                pane ? (pane.id === "mesh_pane" ? "mesh" : "geometry") : "meanline";
            const section = wrapper.closest("details");
            const summary = section && section.querySelector(":scope > summary");
            const context = [workspace[0].toUpperCase() + workspace.slice(1)];
            if (stage !== undefined && stage !== null) context.push(stageElement && stageElement.dataset.stageName || "Stage " + (Number(stage) + 1));
            if (row) context.push(row[0].toUpperCase() + row.slice(1));
            if (summary) context.push(summary.textContent.trim());
            const optimizationRow = wrapper.closest(".td-opt-row");
            if (optimizationRow) {
                const selection = optimizationRow.querySelector(".td-opt-row-head .Select-value-label");
                if (selection) context.push(selection.textContent.trim());
            }
            const aliases = wrapper.dataset.parameterAliases || wrapper.dataset.aliases || "";
            index.push({ wrapper, control, identity, label: label.trim(), stage, row, workspace,
                context: context.join(" · "), terms: normalized([label, aliases, identity.key, context.join(" ")].join(" ")) });
        });
        return index;
    }
    function closeSearch() {
        const results = document.getElementById("td-search-results"), input = document.getElementById("td-parameter-search");
        if (results) results.classList.remove("td-search-open");
        if (input) { input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); }
    }
    function highlightResult() {
        const results = document.getElementById("td-search-results"), input = document.getElementById("td-parameter-search");
        results.querySelectorAll(".td-search-result").forEach((button, index) => {
            button.setAttribute("aria-selected", String(index === selectedResult));
        });
        const selected = results.querySelector('[aria-selected="true"]');
        if (selected) { input.setAttribute("aria-activedescendant", selected.id); selected.scrollIntoView({ block: "nearest" }); }
        else input.removeAttribute("aria-activedescendant");
    }
    function search(query) {
        const results = document.getElementById("td-search-results"), input = document.getElementById("td-parameter-search");
        if (!results || !input) return;
        const tokens = normalized(query).split(" ").filter(Boolean);
        if (!tokens.length) { matches = []; closeSearch(); return; }
        matches = indexFields().filter(item => tokens.every(token => item.terms.includes(token))).slice(0, 30);
        selectedResult = 0;
        results.replaceChildren();
        if (!matches.length) {
            const empty = document.createElement("div"); empty.className = "td-search-empty";
            empty.textContent = "No matching parameters"; empty.setAttribute("role", "status"); results.appendChild(empty);
        }
        matches.forEach((item, index) => {
            const button = document.createElement("button");
            button.type = "button"; button.className = "td-search-result"; button.id = "td-parameter-result-" + index;
            button.setAttribute("role", "option"); button.setAttribute("tabindex", "-1");
            const title = document.createElement("strong"), subtitle = document.createElement("small");
            title.textContent = item.label; subtitle.textContent = item.context; button.append(title, subtitle);
            button.addEventListener("mousedown", event => event.preventDefault());
            button.addEventListener("click", () => reveal(item)); results.appendChild(button);
        });
        results.classList.add("td-search-open"); input.setAttribute("aria-expanded", "true"); highlightResult();
    }
    function reveal(item) {
        closeSearch(); setWorkspace(item.workspace);
        if (window.innerWidth <= 850) setDrawer(true);
        if (item.stage !== undefined && item.stage !== null) {
            root.dataset.stageIndex = String(item.stage);
            setProps("td-stage-selector", { value: Number(item.stage) });
            if (item.workspace !== "meanline") setProps(item.workspace + "_stage", { value: Number(item.stage) });
        }
        if (item.row) {
            root.dataset.bladeRow = item.row;
            setProps("td-row-selector", { value: item.row });
            if (item.workspace !== "meanline") setProps(item.workspace + "_row", { value: item.row });
        }
        applyScope();
        const openParents = () => {
            for (let node = item.wrapper.parentElement; node && node !== root; node = node.parentElement) {
                if (node.tagName === "DETAILS") node.open = true;
            }
        };
        if (item.wrapper.closest("#opt-panel")) {
            const target = item.identity.type || (item.control && item.control.id) || "";
            const tab = target.includes("variable") ? "variables" : target.includes("constraint") || target.includes("comparison") ? "constraints" : "settings";
            setProps("opt-tabs", { active_tab: tab });
            if (item.wrapper.closest("#opt-expert-tolerances,#opt-expert-steps")) {
                setProps("opt-expert-settings", { value: true });
            }
        }
        openParents();
        window.setTimeout(() => {
            openParents();
            const input = (item.control && item.control.id && document.getElementById(item.control.id)) || item.control ||
                item.wrapper.querySelector("input,select,textarea,[tabindex]");
            const wrapper = input && input.closest("[data-parameter-label],.td-field,.td-parameter,.td-opt-field") || item.wrapper;
            wrapper.scrollIntoView({ block: "center", behavior: "instant" });
            if (input) input.focus({ preventScroll: true });
            wrapper.classList.add("td-found-field");
            window.setTimeout(() => wrapper.classList.remove("td-found-field"), 1700);
        }, 100);
    }
    function setupResize() {
        const handle = document.getElementById("td-inspector-resize");
        if (!handle) return;
        const stored = Number(readPreference(WIDTH_KEY));
        const width = value => Math.max(290, Math.min(480, window.innerWidth - 410, value));
        const setWidth = value => {
            const next = width(value); root.style.setProperty("--td-inspector-width", next + "px");
            handle.setAttribute("aria-valuenow", String(Math.round(next))); return next;
        };
        if (stored >= 290 && stored <= 480) setWidth(stored);
        handle.setAttribute("aria-valuemin", "290"); handle.setAttribute("aria-valuemax", "480");
        handle.setAttribute("aria-valuenow", String(Math.round(document.getElementById("design-controls").getBoundingClientRect().width)));
        handle.addEventListener("pointerdown", event => {
            if (event.button !== 0) return;
            const start = event.clientX, initial = document.getElementById("design-controls").getBoundingClientRect().width;
            root.classList.add("td-resizing"); handle.setPointerCapture(event.pointerId);
            const move = next => setWidth(initial + next.clientX - start);
            const finish = () => {
                root.classList.remove("td-resizing"); handle.removeEventListener("pointermove", move);
                handle.removeEventListener("pointerup", finish); handle.removeEventListener("pointercancel", finish);
                savePreference(WIDTH_KEY, String(document.getElementById("design-controls").getBoundingClientRect().width));
                window.dispatchEvent(new Event("resize"));
            };
            handle.addEventListener("pointermove", move); handle.addEventListener("pointerup", finish); handle.addEventListener("pointercancel", finish);
        });
        handle.addEventListener("keydown", event => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
            event.preventDefault();
            let next = document.getElementById("design-controls").getBoundingClientRect().width;
            next = event.key === "Home" ? 290 : event.key === "End" ? 480 : next + (event.key === "ArrowLeft" ? -10 : 10);
            savePreference(WIDTH_KEY, String(setWidth(next))); window.dispatchEvent(new Event("resize"));
        });
    }
    function queueRefresh() {
        if (pendingRefresh) return;
        pendingRefresh = true;
        window.setTimeout(() => { pendingRefresh = false; attach(); applyScope(); labelDropdowns(); formatNumericInputs(); themeFigures(); }, 80);
    }
    function attach() {
        const next = document.getElementById("td-workbench");
        if (!next || next === root) return;
        root = next; root.dataset.workspace = root.dataset.workspace || "meanline";
        applyTheme(currentTheme, false); applyScope(); setupResize(); setupHeader();
        formatNumericInputs();
        root.addEventListener("input", event => {
            const input = event.target;
            if (continuousNumber(input) && input.id) {
                if (input.value !== "") typedNumbers.set(input.id, input.value);
                else typedNumbers.delete(input.id);
            }
        }, true);
        root.addEventListener("focusout", event => {
            if (continuousNumber(event.target)) scheduleNumericFormat(event.target);
        }, true);
        root.querySelectorAll(".td-workspace-rail [data-workspace]").forEach(button => {
            button.setAttribute("aria-current", button.dataset.workspace === root.dataset.workspace ? "page" : "false");
        });
        const input = document.getElementById("td-parameter-search"), results = document.getElementById("td-search-results");
        if (input && results) {
            input.setAttribute("role", "combobox"); input.setAttribute("aria-autocomplete", "list");
            input.setAttribute("aria-controls", results.id); input.setAttribute("aria-expanded", "false");
            input.setAttribute("autocomplete", "off"); results.setAttribute("role", "listbox");
            results.setAttribute("aria-label", "Matching parameters");
            input.addEventListener("input", () => search(input.value));
            input.addEventListener("focus", () => { if (input.value.trim()) search(input.value); });
            input.addEventListener("keydown", event => {
                if (event.key === "Escape") { closeSearch(); return; }
                if (!matches.length || !results.classList.contains("td-search-open")) return;
                if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                    event.preventDefault(); selectedResult = (selectedResult + (event.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length; highlightResult();
                } else if (event.key === "Enter") { event.preventDefault(); reveal(matches[selectedResult]); }
            });
        }
        root.addEventListener("click", event => {
            const theme = event.target.closest(".td-theme-toggle [data-theme]");
            if (theme) { applyTheme(theme.dataset.theme, true); return; }
            const nav = event.target.closest("[data-workspace]");
            if (nav && nav !== root && (nav.closest(".td-workspace-rail") || nav.classList.contains("td-brand"))) {
                event.preventDefault(); setWorkspace(nav.dataset.workspace); return;
            }
            if (event.target.closest("#td-collapse-all")) {
                root.querySelectorAll(".td-inspector details").forEach(details => { details.open = false; }); return;
            }
            if (event.target.closest("#td-inspector-toggle")) { setDrawer(!root.classList.contains("td-inspector-open")); return; }
            if (event.target.closest(".td-inspector-backdrop")) { setDrawer(false); return; }
            const editStage = event.target.closest("[data-edit-stage]");
            if (editStage) {
                root.dataset.stageIndex = editStage.dataset.editStage; setProps("td-stage-selector", { value: Number(editStage.dataset.editStage) });
                if (window.innerWidth <= 850) setDrawer(true);
                applyScope();
                const section = root.querySelector("[data-stage-index='" + Number(editStage.dataset.editStage) + "']");
                if (section) {
                    for (let node = section.parentElement; node && node !== root; node = node.parentElement) if (node.tagName === "DETAILS") node.open = true;
                    section.scrollIntoView({ block: "start" });
                }
            }
            if (!event.target.closest(".td-search-wrap") && event.target !== input && !event.target.closest("#td-search-results")) closeSearch();
        });
        root.addEventListener("toggle", event => {
            const details = event.target;
            if (details.matches("details.td-section[id]")) setProps(details.id, { open: details.open });
            if (details.matches("details.td-results-disclosure") && details.open) {
                requestAnimationFrame(() => {
                    if (window.Plotly) details.querySelectorAll(".js-plotly-plot").forEach(graph => {
                        if (graph._fullLayout) window.Plotly.Plots.resize(graph);
                    });
                });
            }
        }, true);
        new MutationObserver(records => {
            if (records.some(record => ["data-stage-index", "data-blade-row", "data-workspace"].includes(record.attributeName))) applyScope();
            if (records.some(record => record.attributeName === "data-design-revision")) {
                typedNumbers.clear();
                const preserveOptimization = root.dataset.preserveOptimizationPanel === "true";
                root.querySelectorAll(".td-inspector details").forEach(details => {
                    if (!preserveOptimization || !details.closest("#opt-panel")) details.open = false;
                });
                closeSearch();
                formatNumericInputs();
            }
        }).observe(root, { attributes: true });
    }
    document.addEventListener("keydown", event => {
        if (!root) return;
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
            event.preventDefault(); if (window.innerWidth <= 850) setDrawer(true);
            const input = document.getElementById("td-parameter-search"); if (input) { input.focus(); input.select(); }
        }
        if (event.key === "Escape") {
            const results = document.getElementById("td-search-results");
            if (results && results.classList.contains("td-search-open")) closeSearch();
            else if (root.classList.contains("td-inspector-open")) { setDrawer(false); document.getElementById("td-inspector-toggle").focus(); }
        }
    });
    media.addEventListener("change", event => { if (!explicitTheme) applyTheme(event.matches ? "dark" : "light", false); });
    new MutationObserver(() => { queueThemeRefresh(); queueRefresh(); })
        .observe(document.documentElement, { childList: true, subtree: true });
    window.TurbodashWorkbench = { applyTheme, setWorkspace, refresh: queueRefresh, indexFields };
    attach();
})();
