/* Local documentation interactions only; no app state or calculations. */
(function () {
  "use strict";
  const root = document.body;
  const storageKey = "turbodash.design-blueprint.theme";
  const preference = window.matchMedia("(prefers-color-scheme: dark)");
  const choices = document.querySelectorAll('input[name="guide-theme"]');
  let explicit = null;
  try {
    const stored = window.localStorage.getItem(storageKey);
    if (stored === "light" || stored === "dark") explicit = stored;
  } catch (_) { /* Local files or private browsing may disallow storage. */ }

  function applyTheme(theme) {
    root.dataset.theme = theme;
    choices.forEach(choice => { choice.checked = choice.value === theme; });
  }
  applyTheme(explicit || (preference.matches ? "dark" : "light"));
  choices.forEach(choice => choice.addEventListener("change", () => {
    if (!choice.checked) return;
    explicit = choice.value;
    applyTheme(explicit);
    try { window.localStorage.setItem(storageKey, explicit); }
    catch (_) { /* The selected theme still applies for this page view. */ }
  }));
  const followPreference = event => {
    if (!explicit) applyTheme(event.matches ? "dark" : "light");
  };
  if (preference.addEventListener) preference.addEventListener("change", followPreference);
  else preference.addListener(followPreference);

  const group = document.querySelector('[data-tree-group="stage-one"]');
  const leaves = [...document.querySelectorAll('[data-tree-leaf="stage-one"]')];
  const status = document.getElementById("tree-status");
  function updateSelection() {
    const count = leaves.filter(leaf => leaf.checked).length;
    group.checked = count === leaves.length;
    group.indeterminate = count > 0 && count < leaves.length;
    status.textContent = count + " of " + leaves.length + " sample quantities selected. Documentation only.";
  }
  group.addEventListener("change", () => {
    leaves.forEach(leaf => { leaf.checked = group.checked; });
    updateSelection();
  });
  leaves.forEach(leaf => leaf.addEventListener("change", updateSelection));
  updateSelection();
}());
