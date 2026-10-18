"""Plot solved assembly layouts."""
import numpy as np


def plot_assembly_layout(layout):
    """Return a Matplotlib figure and axes showing endwalls and row envelopes."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axial = np.array([point["x"] for point in layout["endwalls"]])
    hub = np.array([point["hub_radius"] for point in layout["endwalls"]])
    shroud = np.array([point["shroud_radius"] for point in layout["endwalls"]])
    axes.plot(axial, hub, color="black", label="Hub / shroud")
    axes.plot(axial, shroud, color="black")
    for row in layout["rows"]:
        row_axial = np.unique(np.concatenate((
            [row["envelope_x_min"], row["envelope_x_max"]],
            axial[(axial > row["envelope_x_min"]) & (axial < row["envelope_x_max"])],
        )))
        axes.fill_between(row_axial, np.interp(row_axial, axial, hub),
                          np.interp(row_axial, axial, shroud), alpha=0.3,
                          color="tab:blue" if row["blade_row"] == "stator" else "tab:orange")
        midpoint = 0.5 * (row["envelope_x_min"] + row["envelope_x_max"])
        axes.text(midpoint, np.interp(midpoint, axial, shroud),
                  f"S{row['stage_number']} {row['blade_row']}", ha="center", va="bottom")
    for index, interface in enumerate(layout["interfaces"]):
        axes.plot([interface["x"], interface["x"]],
                  [interface["hub_radius"], interface["shroud_radius"]], "r--",
                  label="Row interface" if index == 0 else None)
    axes.set(xlabel="Axial coordinate [m]", ylabel="Radius [m]",
             title="Proposed assembly: shaded row envelopes, not blade surfaces")
    axes.set_aspect("equal", adjustable="datalim")
    axes.legend()
    return figure, axes

