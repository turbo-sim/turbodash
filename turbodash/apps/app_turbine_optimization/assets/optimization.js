// Follow new convergence output unless the reader has scrolled up.
(function () {
    let terminal = null;
    let follow = true;
    const attach = () => {
        const next = document.getElementById('opt-log');
        if (!next || next === terminal) return;
        terminal = next;
        terminal.addEventListener('scroll', () => {
            follow = terminal.scrollHeight - terminal.scrollTop - terminal.clientHeight < 45;
        });
        new MutationObserver(() => {
            if (follow) requestAnimationFrame(() => { terminal.scrollTop = terminal.scrollHeight; });
        }).observe(terminal, {childList: true, subtree: true, characterData: true});
        document.getElementById('opt-terminal').addEventListener('toggle', () => {
            if (follow) terminal.scrollTop = terminal.scrollHeight;
        });
    };
    new MutationObserver(attach).observe(document.documentElement, {childList: true, subtree: true});
    attach();
})();

// Hierarchical pickers keep a cumulative selection journal. Each callback can
// replay every action newer than its acknowledged revision, including clicks
// made while another request is still in flight.
(function () {
    let panel = null, scheduled = false, revision = 0, operations = [];
    let lastFocused = null;
    const branches = new Map(), scrollPositions = new Map();
    const parse = (value, fallback) => {
        try { return JSON.parse(value); } catch (_) { return fallback; }
    };
    const memberMatches = (member, spec, kind) => member.path === spec.path &&
        (kind === "variable" || member.comparison === spec.comparison);
    function membersFor(operation) {
        if (!operation.group) return [];
        const button = Array.from(panel.querySelectorAll("[data-opt-group]")).find(
            item => item.dataset.optKind === operation.kind && item.dataset.optGroup === operation.group);
        return button ? parse(button.dataset.optMembers, []) : [];
    }
    function applyOperation(selected, operation) {
        if (operation.row) return selected.filter(member => member.row !== operation.row);
        if (operation.path) {
            if (!operation.selected) return selected.filter(member => member.path !== operation.path);
            if (!selected.some(member => member.path === operation.path)) {
                const button = Array.from(panel.querySelectorAll("[data-opt-path]")).find(
                    item => item.dataset.optKind === operation.kind && item.dataset.optPath === operation.path);
                selected.push({path: operation.path, comparison: button?.dataset.optComparison});
            }
            return selected;
        }
        const members = membersFor(operation);
        if (!operation.selected) {
            return selected.filter(member => !members.some(spec => memberMatches(member, spec, operation.kind)));
        }
        members.forEach(spec => {
            if (!selected.some(member => memberMatches(member, spec, operation.kind))) selected.push({...spec});
        });
        return selected;
    }
    function syncChecks() {
        if (!panel) return;
        const snapshots = {};
        panel.querySelectorAll(".td-opt-picker-tree[data-opt-kind]").forEach(tree => {
            const kind = tree.dataset.optKind;
            const applied = Number(tree.dataset.optAppliedRevision || 0);
            if (!snapshots[kind] || applied >= snapshots[kind].revision) {
                snapshots[kind] = {revision: applied, selected: parse(tree.dataset.optSelection, [])};
            }
        });
        Object.entries(snapshots).forEach(([kind, snapshot]) => {
            operations.filter(op => op.kind === kind && op.revision > snapshot.revision)
                .forEach(op => { snapshot.selected = applyOperation(snapshot.selected, op); });
        });
        const busy = panel.classList.contains("td-opt-busy");
        panel.querySelectorAll(".td-opt-picker-item").forEach(button => {
            const kind = button.dataset.optKind;
            const selected = snapshots[kind]?.selected || [];
            let state;
            if (button.dataset.optGroup) {
                const members = parse(button.dataset.optMembers, []);
                const count = members.filter(spec => selected.some(member => memberMatches(member, spec, kind))).length;
                state = count === 0 ? "false" : count === members.length ? "true" : "mixed";
            } else {
                state = selected.some(member => member.path === button.dataset.optPath) ? "true" : "false";
            }
            button.setAttribute("aria-checked", state);
            button.disabled = busy;
        });
        if (snapshots.variable && snapshots.constraint) {
            const acknowledged = Math.min(snapshots.variable.revision, snapshots.constraint.revision);
            operations = operations.filter(op => op.revision > acknowledged);
            revision = Math.max(revision, acknowledged);
        }
    }
    function publish(operation) {
        operations.push({...operation, revision: ++revision});
        syncChecks();
        window.dash_clientside?.set_props("opt-selection-request", {
            data: {revision, operations: operations.map(op => ({...op}))}
        });
    }
    function filterMenu(menu) {
        const query = (menu.querySelector(".td-opt-picker-filter")?.value || "")
            .toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
        menu.querySelectorAll(".td-opt-picker-item").forEach(button => {
            const text = (button.dataset.optSearch || button.textContent).toLocaleLowerCase();
            button.hidden = !query.every(word => text.includes(word));
        });
        Array.from(menu.querySelectorAll(".td-opt-picker-branch")).reverse().forEach(branch => {
            branch.hidden = !Array.from(branch.querySelectorAll(".td-opt-picker-item")).some(button => !button.hidden);
            const remembered = branches.get(branch.dataset.optNode);
            if (query.length && !branch.hidden) branch.open = true;
            else if (!query.length && remembered !== undefined) branch.open = remembered;
        });
        const empty = menu.querySelector(".td-opt-picker-no-match");
        if (empty) empty.hidden = !!menu.querySelector(".td-opt-picker-item:not([hidden])");
    }
    function revealMenu(details) {
        if (!details?.open) return;
        const menu = details.querySelector(".td-opt-picker-menu");
        const inspector = details.closest(".td-inspector-scroll");
        if (!menu || !inspector) return;
        const heading = panel.querySelector(":scope > summary")?.getBoundingClientRect().height || 0;
        const actions = details.closest(".td-opt-picker-actions")?.getBoundingClientRect().height || 0;
        menu.style.maxHeight = `${Math.max(40, Math.min(360, window.innerHeight * .45,
            inspector.clientHeight - heading - actions - 20))}px`;
        const bounds = menu.getBoundingClientRect(), viewport = inspector.getBoundingClientRect();
        if (bounds.bottom > viewport.bottom - 6) {
            inspector.scrollTop += bounds.bottom - viewport.bottom + 6;
        } else if (bounds.top < viewport.top + heading + 6) {
            inspector.scrollTop -= viewport.top + heading + 6 - bounds.top;
        }
    }
    function refresh() {
        scheduled = false;
        if (!panel) return;
        panel.querySelectorAll(".td-opt-picker-menu").forEach(menu => {
            filterMenu(menu);
            const identity = menu.closest("details")?.id;
            if (scrollPositions.has(identity)) menu.scrollTop = scrollPositions.get(identity);
        });
        syncChecks();
        panel.querySelectorAll("details.td-opt-picker[open]").forEach(revealMenu);
        if (lastFocused && !lastFocused.element.isConnected &&
            (document.activeElement === document.body || !document.activeElement)) {
            const replacement = Array.from(panel.querySelectorAll(".td-opt-picker-item")).find(button =>
                button.dataset.optKind === lastFocused.kind &&
                button.dataset.optPath === lastFocused.path && button.dataset.optGroup === lastFocused.group);
            if (replacement?.closest("details[id]")?.open && !replacement.hidden) {
                replacement.focus({preventScroll: true});
                lastFocused.element = replacement;
            }
        }
    }
    function schedule() {
        if (!scheduled) { scheduled = true; requestAnimationFrame(refresh); }
    }
    function closeMenus(except, restoreFocus = false) {
        panel?.querySelectorAll("details.td-opt-picker").forEach(menu => {
            if (menu !== except && menu.open) {
                menu.open = false;
                if (restoreFocus) menu.querySelector(":scope > summary")?.focus({preventScroll: true});
            }
        });
    }
    function attach() {
        const next = document.getElementById("opt-panel");
        if (!next || next === panel) return;
        panel = next;
        panel.addEventListener("click", event => {
            if (panel.classList.contains("td-opt-busy")) return;
            const button = event.target.closest(".td-opt-picker-item,.td-opt-remove");
            if (!button) return;
            if (button.classList.contains("td-opt-remove")) {
                publish({kind: button.dataset.optKind, row: button.dataset.optRow, selected: false});
                return;
            }
            const operation = {kind: button.dataset.optKind, selected: button.getAttribute("aria-checked") !== "true"};
            if (button.dataset.optGroup) operation.group = button.dataset.optGroup;
            else operation.path = button.dataset.optPath;
            publish(operation);
        });
        panel.addEventListener("input", event => {
            if (event.target.matches(".td-opt-picker-filter")) {
                filterMenu(event.target.closest(".td-opt-picker-menu"));
            }
        });
        panel.addEventListener("focusin", event => {
            const button = event.target.closest(".td-opt-picker-item");
            if (button) lastFocused = {element: button, kind: button.dataset.optKind,
                path: button.dataset.optPath, group: button.dataset.optGroup};
            else lastFocused = null;
        });
        panel.addEventListener("toggle", event => {
            const details = event.target;
            if (details.matches(".td-opt-picker")) {
                const summary = details.querySelector(":scope > summary");
                summary?.setAttribute("aria-expanded", String(details.open));
                if (details.open) {
                    closeMenus(details);
                    requestAnimationFrame(() => {
                        details.querySelector(".td-opt-picker-filter")?.focus({preventScroll: true});
                        revealMenu(details);
                    });
                }
            } else if (details.matches(".td-opt-picker-branch")) {
                const query = details.closest(".td-opt-picker-menu")?.querySelector(".td-opt-picker-filter")?.value;
                if (!query) branches.set(details.dataset.optNode, details.open);
                requestAnimationFrame(() => revealMenu(details.closest("details.td-opt-picker")));
            }
        }, true);
        panel.addEventListener("scroll", event => {
            if (event.target.matches(".td-opt-picker-menu")) {
                scrollPositions.set(event.target.closest("details")?.id, event.target.scrollTop);
            }
        }, true);
        new MutationObserver(schedule).observe(panel, {childList: true, subtree: true});
        new MutationObserver(schedule).observe(panel, {attributes: true, attributeFilter: ["class"]});
        schedule();
    }
    document.addEventListener("click", event => {
        if (!event.target.closest(".td-opt-picker")) closeMenus(null);
    });
    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && panel?.querySelector("details.td-opt-picker[open]")) {
            event.preventDefault();
            event.stopImmediatePropagation();
            closeMenus(null, true);
        }
    }, true);
    window.addEventListener("resize", () => {
        panel?.querySelectorAll("details.td-opt-picker[open]").forEach(revealMenu);
    });
    new MutationObserver(attach).observe(document.documentElement, {childList: true, subtree: true});
    window.TurbodashOptimization = {selectionRevision: () => revision};
    attach();
})();
