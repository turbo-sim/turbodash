// Host transport only. Native nurbspy owns the multi-object viewport, dragging,
// picking, numeric controls, accepted float64 models, and undo/redo history.
let current = null;
let rendered = null;
let mounting = Promise.resolve();

function coordinatePolicy(root) {
    // Native controls remain responsible for edits. Suppress an untouched
    // field's Enter/blur so its rounded display string never alters float64 data.
    const initial = event => {
        if (event.target.matches('.nrb-coordinate')) event.target.dataset.hostValue = event.target.value;
    };
    const unchanged = event => {
        const field = event.target;
        if (!field.matches('.nrb-coordinate') || (event.type === 'keydown' && event.key !== 'Enter')) return;
        if (field.value === field.dataset.hostValue) {
            event.preventDefault(); event.stopImmediatePropagation();
        } else field.dataset.hostValue = field.value;
    };
    const commitBeforePicking = event => {
        const field = document.activeElement;
        if (event.target.tagName === 'CANVAS' && root.contains(field) && field.matches('.nrb-coordinate')) field.blur();
    };
    root.addEventListener('focusin', initial);
    root.addEventListener('keydown', unchanged, true);
    root.addEventListener('change', unchanged, true);
    root.addEventListener('pointerdown', commitBeforePicking, true);
    return () => {
        root.removeEventListener('focusin', initial);
        root.removeEventListener('keydown', unchanged, true);
        root.removeEventListener('change', unchanged, true);
        root.removeEventListener('pointerdown', commitBeforePicking, true);
    };
}

async function post(base, path, body) {
    const response = await fetch(base + path, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Wall evaluation failed.');
    return result;
}

export function mount(scene, base, isCurrent = () => true) {
    if (!isCurrent()) return Promise.resolve({superseded: true});
    const session = {identity: scene.identity, config: scene.host_config,
        state: scene.state, controller: null, isCurrent};
    current = session;
    window.dash_clientside.set_props('wall-apply', {disabled: true});
    // Dash may deliver several geometry generations while native mounting is
    // awaiting its first frame. Serialize that lifecycle and skip superseded
    // generations so the visible Figure and captured controller stay identical.
    const result = mounting.then(() => mountSession(session, scene, base));
    mounting = result.catch(() => {});
    return result;
}

async function mountSession(session, scene, base) {
    if (current !== session || !session.isCurrent()) return {superseded: true};
    rendered?.cleanup?.();
    rendered?.controller?.dispose();
    rendered = null;
    try {
        const {mountFigure} = await import(base + '/assets/figure.js');
        if (current !== session || !session.isCurrent()) return {superseded: true};
        session.controller = await mountFigure('wall-editor-viewport', scene, {
            inspect: (name, packet) => post(base, '/inspect', {config: session.config, name, packet}),
            evaluate: async (name, packet) => {
                const result = await post(base, '/evaluate', {
                    config: session.config, state: session.state, name, packet,
                });
                const bytes = Uint8Array.from(atob(result.vertices), c => c.charCodeAt(0));
                return {...result, positions: new Float32Array(bytes.buffer)};
            },
            scenes: async state => {
                session.state = state;
                return post(base, '/scenes', {config: session.config, state});
            },
            onState: state => {
                if (current !== session || !session.isCurrent()) return;
                session.state = state;
                window.dash_clientside.set_props('wall-editor-state', {data: state});
            },
        });
        if (current !== session || !session.isCurrent()) {
            session.controller.dispose();
            return {superseded: true};
        }
        rendered = session;
        session.cleanup = coordinatePolicy(document.getElementById('wall-editor-viewport'));
        if (current === session && session.isCurrent()) window.dash_clientside.set_props('wall-apply', {disabled: false});
        return {mounted: true, identity: scene.identity};
    } catch (error) {
        if (current === session && session.isCurrent()) window.dash_clientside.set_props('wall-notice', {
            children: 'Editor unavailable: ' + error.message,
        });
        return {error: error.message};
    }
}

export async function capture() {
    const session = current;
    if (!session?.controller) return {error: 'Wait for the wall editor to load.'};
    if (!session.isCurrent()) return {error: 'Engineering inputs changed. Wait for the refreshed editor before applying.'};
    window.dash_clientside.set_props('wall-apply', {disabled: true});
    try {
        await session.controller.ready();
        if (current !== session || !session.isCurrent()) throw new Error('Engineering inputs changed. Apply after the editor refreshes.');
        return {identity: session.identity, state: session.controller.snapshot(), nonce: Date.now()};
    } catch (error) {
        return {error: error.message, nonce: Date.now()};
    } finally {
        if (current === session && session.isCurrent()) window.dash_clientside.set_props('wall-apply', {disabled: false});
    }
}
