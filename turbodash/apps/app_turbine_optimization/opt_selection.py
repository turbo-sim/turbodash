"""Manage the hierarchical selectors for optimization variables and constraints.

This module builds the searchable menus that organize choices by overall
turbine, stage, and blade row. Users can select individual quantities or
predefined groups, with checkbox states reflecting the current selection.

Selection changes are applied in their original order so that rapid clicks
and delayed browser callbacks preserve the user's choices. Existing rows keep
their identifiers and edited values, duplicate entries are avoided, and lower
and upper constraints on the same quantity remain separate editable rows.
"""

import json
from uuid import uuid4

from dash import dcc, html


def row_identity(row, kind):
    """Treat opposite constraint bounds as separate configured quantities."""
    return (row.get("path"), row.get("comparison", "=")) if kind == "constraint" else row.get("path")


def group_members(group, kind):
    if kind == "constraint" and group.get("rows"):
        return [dict(row) for row in group["rows"]]
    return [dict(path=path) for path in group.get("paths", [])]


def selected_members(rows, kind):
    return [
        dict(path=row["path"], row=row["id"],
             **({"comparison": row.get("comparison", "=")} if kind == "constraint" else {}))
        for row in rows if row.get("path")
    ]


def apply_selection_operations(rows, catalog, groups, kind, operations):
    """Replay actions in order while retaining existing IDs and numeric edits."""
    rows = [dict(row) for row in rows]
    for operation in operations:
        if operation.get("kind") != kind:
            continue
        if operation.get("row"):
            if operation.get("selected") is False:
                rows = [row for row in rows if row["id"] != operation["row"]]
            continue
        group = groups.get(operation.get("group"))
        if operation.get("group") and group is None:
            continue
        members = group_members(group, kind) if group else [dict(path=operation.get("path"))]
        members = [member for member in members if member.get("path") in catalog]
        if operation.get("selected") is False:
            if group:
                identities = {row_identity(member, kind) for member in members}
                rows = [row for row in rows if row_identity(row, kind) not in identities]
            else:
                paths = {member["path"] for member in members}
                rows = [row for row in rows if row.get("path") not in paths]
            continue
        if operation.get("selected") is not True:
            continue
        for member in members:
            path = member["path"]
            choice = catalog[path]
            if kind == "variable":
                defaults = dict(lower=choice.get("lower"), initial=None, upper=choice.get("upper"))
            else:
                defaults = dict(comparison=choice.get("comparison", "="),
                                target=choice.get("target", 0), scale=choice.get("scale", 1))
            defaults.update({key: value for key, value in member.items() if key != "path"})
            candidate = dict(id=uuid4().hex, path=path, label=choice.get("label"), **defaults)
            # An individual constraint checkbox represents every bound on its
            # path. Presets distinguish lower/upper bounds by comparison.
            exists = any(
                row_identity(row, kind) == row_identity(candidate, kind)
                if group or kind == "variable" else row.get("path") == path
                for row in rows
            )
            if not exists:
                rows.append(candidate)
    return rows


def selector_menu(kind, groups=False):
    suffix = "-group" if groups else ""
    label = "+ Add groups" if groups else f"+ Add {kind}"
    return html.Details([
        html.Summary(label, id=f"opt-add-{kind}{suffix}", n_clicks=0,
                     className="td-opt-picker-toggle", role="button"),
        html.Div([
            html.Label([
                html.Span(f"Find {kind} {'groups' if groups else 'quantities'}", className="visually-hidden"),
                dcc.Input(id=f"opt-{kind}{suffix}-filter", type="search",
                          placeholder="Find a group…" if groups else f"Find a {kind}…",
                          className="td-opt-picker-filter td-opt-picker-search"),
            ]),
            html.Div(id=f"opt-{kind}{suffix}-picker"),
            html.P("No matching quantities." if not groups else "No matching groups.",
                   className="td-opt-picker-no-match", hidden=True),
        ], className="td-opt-picker-menu"),
    ], id=f"opt-{kind}{suffix}-menu", className="td-opt-picker",
        **{"data-opt-menu": f"{kind}-groups" if groups else kind})


def _button(kind, key, choice, state, members=None):
    group = members is not None
    label = choice.get("short_label", choice["label"])
    description = choice.get("description", "")
    attributes = {
        "role": "checkbox", "aria-checked": state,
        "aria-label": choice["label"], "data-opt-kind": kind,
        "data-opt-group" if group else "data-opt-path": key,
        "data-opt-search": " ".join([choice["label"], *choice.get("hierarchy", []), key, description]),
    }
    if group:
        attributes["data-opt-members"] = json.dumps([
            dict(path=member["path"],
                 **({"comparison": member.get("comparison", "=")} if kind == "constraint" else {}))
            for member in members
        ])
    elif kind == "constraint":
        attributes["data-opt-comparison"] = choice.get("comparison", "=")
    return html.Button([
        html.Span(className="td-opt-picker-check", **{"aria-hidden": "true"}),
        html.Span([html.Span(label), html.Small(description, className="td-opt-picker-description")]
                  if description else label, className="td-opt-picker-label"),
    ], type="button", className="td-opt-picker-item", key=f"{kind}:{'group' if group else 'path'}:{key}",
        title=description or choice["label"], **attributes)


def picker_tree(catalog, rows, kind, revision, groups=None):
    """Render labelled, keyboard-operable branches with tri-state group checks."""
    tree = {}
    selected = {row_identity(row, kind) for row in rows}
    selected_paths = {row.get("path") for row in rows}
    for key, choice in (groups if groups is not None else catalog).items():
        members = group_members(choice, kind) if groups is not None else None
        if members is not None:
            identities = {row_identity(member, kind) for member in members}
            matched = len(identities & selected)
            checked = "true" if identities and matched == len(identities) else "mixed" if matched else "false"
        else:
            checked = "true" if key in selected_paths else "false"
        node = tree
        for label in choice.get("hierarchy", ["Presets" if groups is not None else "Overall"]):
            node = node.setdefault(label, {})
        node.setdefault(None, []).append(_button(kind, key, choice, checked, members))

    def branches(node, ancestors=()):
        children = list(node.get(None, []))
        for label, branch in node.items():
            if label is None:
                continue
            identity = ":".join([kind, "groups" if groups is not None else "quantities", *ancestors, label])
            children.append(html.Details([
                html.Summary(label),
                html.Div(branches(branch, (*ancestors, label)), className="td-opt-picker-children"),
            ], open=True, className="td-opt-picker-branch", key=identity,
                **{"data-opt-branch": identity, "data-opt-node": identity}))
        return children

    return html.Div(branches(tree), className="td-opt-picker-tree", **{
        "data-opt-kind": kind,
        "data-opt-selection": json.dumps(selected_members(rows, kind)),
        "data-opt-applied-revision": revision,
    })


def picker_signature(catalog, editor):
    """Numeric-only edits leave the open selector DOM, focus and scroll intact."""
    selection = {
        kind: sorted((row.get("path"), row.get("comparison", "") if kind == "constraints" else "", row["id"])
                     for row in editor.get(kind, []))
        for kind in ("variables", "constraints")
    }
    metadata = {
        kind: {
            path: {key: value for key, value in choice.items()
                   if key in ("label", "short_label", "hierarchy", "description")}
            for path, choice in catalog.get(kind, {}).items()
        }
        for kind in ("variables", "constraints")
    }
    return json.dumps(dict(
        metadata=metadata, selection=selection,
        groups=catalog.get("groups", {}),
        revision=editor.get("selection_revision", 0),
    ), sort_keys=True)
