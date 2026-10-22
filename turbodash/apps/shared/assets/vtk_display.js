/* Commit keyboard display-copy edits from the value shown by the slider. */
(function () {
    "use strict";
    const adjustmentKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
                            "Home", "End", "PageUp", "PageDown"];
    document.addEventListener("keyup", function (event) {
        if (!adjustmentKeys.includes(event.key)) return;
        const handle = event.target.closest('[role="slider"]');
        const host = handle && handle.closest("[id]");
        if (!host) return;
        let identity;
        try {identity = JSON.parse(host.id);} catch (_) {return;}
        if (!identity || identity.type !== "vtk-row-slider") return;
        const value = Number(handle.getAttribute("aria-valuenow"));
        if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) return;
        // rc-slider's mouseup mode may publish the previous keyboard value
        // before its handle updates. Key release sees the settled value and
        // follows the ordinary count/slider callback. Mouse dragging retains
        // its existing release-only update, and focus alone commits nothing.
        window.dash_clientside.set_props(identity, {value: value});
    }, true);
}());
