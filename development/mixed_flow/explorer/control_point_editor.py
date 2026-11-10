"""Compatibility host for the shared nurbspy wall editor."""

from turbodash.apps.shared.wall_editor import (
    EDITOR_ID,
    WALLS,
    WallEditor,
    nurbspy_api,
    _curve,
)
from . import model

_editor = WallEditor(model)
identity = _editor.identity
make_editor = _editor.make_editor
make_scene = _editor.make_scene
config_from_state = _editor.config_from_state
evaluate_edit = _editor.evaluate_edit
register = _editor.register
