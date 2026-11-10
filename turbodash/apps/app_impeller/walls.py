"""Impeller configuration policy for the shared native wall editor.

Only the channel is constructed during local point dragging. The app applies
the returned configuration through its full blade/disk validation pipeline.
"""

from copy import deepcopy

from turbodash.apps.shared.wall_editor import WallEditor, nurbspy_api
from turbodash.geometry.impeller import (
    MeanlineDimensions,
    MeridionalChannel,
    WallShape,
    normalize_config,
)


class _ChannelModel:
    normalize_config = staticmethod(normalize_config)

    @staticmethod
    def build_geometry(config):
        config = normalize_config(config)
        channel = MeridionalChannel(
            MeanlineDimensions(**config["dimensions"]),
            WallShape(**config["hub"]),
            WallShape(**config["shroud"]),
            control_point_displacements=config["control_point_displacements"],
        )
        return config, channel, None, None


_editor = WallEditor(_ChannelModel, height=510)
identity = _editor.identity
make_scene = _editor.make_scene
config_from_state = _editor.config_from_state
evaluate_edit = _editor.evaluate_edit


def apply_commit(config, commit=None, *, clear=False):
    """Validate an editor snapshot against its current accepted configuration.

    This returns a candidate and does not publish state. The parent app remains
    the sole writer of accepted geometry, including when a channel passes its
    own checks but the resulting blade does not.
    """
    if clear:
        candidate = deepcopy(config)
        candidate["control_point_displacements"] = {}
        return _ChannelModel.build_geometry(candidate)[0]
    if not isinstance(commit, dict):
        raise ValueError("Wait for the wall editor to load before applying edits.")
    if commit.get("error"):
        raise ValueError(str(commit["error"]))
    if commit.get("identity") != identity(config):
        raise ValueError(
            "Engineering inputs changed. Wait for the refreshed editor and apply again."
        )
    if not isinstance(commit.get("state"), dict):
        raise ValueError("The wall editor snapshot is missing its geometry state.")
    return config_from_state(config, commit["state"])


def register(app, config):
    """Return native editor layout and register its local preview transport.

    Host layout must provide draft-config, wall-commit, wall-notice, wall-apply
    and wall-clear. The host owns wall-commit/wall-clear application callbacks.
    """
    return _editor.register(
        app,
        config,
        base="/_impeller_walls",
        manage_commit=False,
        refresh_trigger="impeller-revision",
    )
