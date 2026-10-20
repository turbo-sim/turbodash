"""Interactions shared by the optional optimization browser regressions."""

from playwright.sync_api import expect


def open_picker(page, kind, *, groups=False):
    suffix = "-group" if groups else ""
    menu = page.locator(f"#opt-{kind}{suffix}-menu")
    if not menu.evaluate("element => element.open"):
        menu.locator(":scope > summary").click()
    expect(menu).to_have_js_property("open", True)
    return menu


def choose_path(page, kind, path, *, selected=True, close=True):
    menu = open_picker(page, kind)
    menu.locator(".td-opt-picker-filter").fill(path)
    checkbox = menu.locator(f'[data-opt-path="{path}"]')
    expect(checkbox).to_be_visible()
    if checkbox.get_attribute("aria-checked") != str(selected).lower():
        checkbox.click()
    expect(checkbox).to_have_attribute("aria-checked", str(selected).lower())
    if close:
        checkbox.press("Escape")


def choose_next(page, kind, *, close=True):
    menu = open_picker(page, kind)
    menu.locator(".td-opt-picker-filter").fill("")
    checkbox = menu.locator('[data-opt-path][aria-checked="false"]').first
    path = checkbox.get_attribute("data-opt-path")
    choose_path(page, kind, path, close=close)
    return path


def choose_group(page, kind, group, *, selected=True, close=True):
    menu = open_picker(page, kind, groups=True)
    menu.locator(".td-opt-picker-filter").fill(group)
    checkbox = menu.locator(f'[data-opt-group="{group}"]')
    expect(checkbox).to_be_visible()
    if checkbox.get_attribute("aria-checked") != str(selected).lower():
        checkbox.click()
    expect(checkbox).to_have_attribute("aria-checked", str(selected).lower())
    if close:
        checkbox.press("Escape")
