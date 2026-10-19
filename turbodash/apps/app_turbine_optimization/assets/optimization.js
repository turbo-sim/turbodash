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
