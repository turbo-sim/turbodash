/* Coalesce pointer movement; all geometry stays in the Python kernel. */
(function () {
    // Dash's slider can commit its step-rounded value on blur even when the
    // user only tabs through it. That stale commit overwrites precise typed
    // values and can reset the next input while it is being edited. Allow a
    // blur commit only after actual pointer/keyboard interaction with a slider.
    const editedSliders = new WeakSet();
    document.addEventListener('pointerdown', function (event) {
        const slider = event.target.closest('.v4-slider');
        if (slider) editedSliders.add(slider);
    }, true);
    document.addEventListener('keydown', function (event) {
        const slider = event.target.closest('.v4-slider');
        if (slider && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
                       'Home', 'End', 'PageUp', 'PageDown'].includes(event.key)) {
            editedSliders.add(slider);
        }
    }, true);
    document.addEventListener('focusout', function (event) {
        const slider = event.target.closest('.v4-slider');
        if (!slider) return;
        if (!editedSliders.has(slider)) event.stopPropagation();
        editedSliders.delete(slider);
    }, true);
    // Dash's debounced number input can retain its last *typed* value after a
    // preset/reset changes its controlled value. Re-entering that old number
    // then emits no value event. Commit the actual DOM number on blur/Enter as
    // well; the Python callback ignores this event when already synchronized.
    function commitNumber(event) {
        const input = event.target;
        // Dash 4 places className on an outer wrapper; inspect the actual
        // input's type/id so this works with both Dash 3 and Dash 4.
        if (!input.matches('input[type="number"]') || !input.id.startsWith('{')) return;
        const id = JSON.parse(input.id);
        if (id.type !== 'parameter') return;
        const value = input.value === '' ? null : Number(input.value);
        window.dash_clientside.set_props('numeric-commit', {data: {
            path: id.path, value: Number.isFinite(value) ? value : null,
            event: performance.now()
        }});
    }
    document.addEventListener('blur', commitNumber, true);
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') commitNumber(event);
    }, true);
    const drag = {active: false, path: null, sentAt: 0, lastValue: null,
                  timer: null, resolve: null};
    function cancelPending() {
        if (drag.timer !== null) clearTimeout(drag.timer);
        if (drag.resolve) drag.resolve(window.dash_clientside.no_update);
        drag.timer = null;
        drag.resolve = null;
    }
    document.addEventListener('pointerdown', function (event) {
        const slider = event.target.closest('.v4-slider');
        if (!slider) return;
        cancelPending();
        drag.active = true;
        drag.path = JSON.parse(slider.id).path;
        drag.sentAt = 0;
        drag.lastValue = null;
    }, true);
    function release() {
        drag.active = false;
        cancelPending();
    }
    document.addEventListener('pointerup', release, true);
    document.addEventListener('pointercancel', release, true);
    window.addEventListener('blur', release);

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        v4: {
            sampleDrag: function (values, identifiers) {
                if (!drag.active) return window.dash_clientside.no_update;
                const index = identifiers.findIndex(item => item.path === drag.path);
                const value = values[index];
                if (value == null || value === drag.lastValue) {
                    cancelPending();
                    return window.dash_clientside.no_update;
                }
                cancelPending();
                function emit() {
                    drag.sentAt = performance.now();
                    drag.lastValue = value;
                    return {path: drag.path, value: value, event: drag.sentAt};
                }
                const delay = Math.max(0, 120 - (performance.now() - drag.sentAt));
                if (!delay) return emit();
                return new Promise(resolve => {
                    drag.resolve = resolve;
                    drag.timer = setTimeout(() => {
                        drag.timer = null;
                        drag.resolve = null;
                        resolve(drag.active ? emit() : window.dash_clientside.no_update);
                    }, delay);
                });
            }
        }
    });
})();
