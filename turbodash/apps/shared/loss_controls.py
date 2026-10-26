"""Loss-model controls shared by the turbine applications."""

from dash import Input, Output, html
import dash_bootstrap_components as dbc

from turbodash.loss_models.options import EXIT_MACH_CORRECTION_MODELS

EXIT_MACH_CORRECTION_HELP = (
    "Applies an additional profile-loss penalty when the blade-row exit Mach "
    "number exceeds 1 (absolute Mach for stators, relative Mach for rotors). "
    "The multiplier is 1 + 60(M_exit - 1)^2 above Mach 1, so it increases "
    "rapidly. This represents conventional converging blading, rather than "
    "converging-diverging blades designed with the method of characteristics. "
    "Leave unchecked if the stator or rotor blades are expected to have a "
    "supersonic design. Unchecked by default (multiplier = 1). "
    "This setting applies to every blade row; other loss corrections remain active."
)


def exit_mach_control(defaults):
    """Build a boolean input that follows the overall-input save/load callbacks."""
    return html.Div(
        [
            html.Div(
                [
                    dbc.Checkbox(
                        id={"scope": "overall", "key": "exit_mach_correction"},
                        value=defaults.get("exit_mach_correction", False),
                        label="Exit Mach number correction",
                    ),
                    html.Span(
                        "?",
                        id="td-exit-mach-help",
                        title=EXIT_MACH_CORRECTION_HELP,
                        className="td-help",
                        tabIndex=0,
                        **{"aria-label": EXIT_MACH_CORRECTION_HELP},
                    ),
                ],
                className="td-field-label",
                style={"gridColumn": "1 / -1"},
            ),
        ],
        id="td-exit-mach-option",
        hidden=exit_mach_option_hidden(defaults.get("loss_model")),
        className="td-field mb-2",
        **{"data-parameter-label": "Exit Mach number correction"},
    )


def exit_mach_option_hidden(model):
    """Show the option only for models that use the shared profile correlation."""
    return model not in EXIT_MACH_CORRECTION_MODELS


def register_loss_callbacks(app):
    """Keep the control mounted so switching models preserves its value."""
    app.callback(
        Output("td-exit-mach-option", "hidden"),
        Input({"scope": "overall", "key": "loss_model"}, "value"),
    )(exit_mach_option_hidden)
