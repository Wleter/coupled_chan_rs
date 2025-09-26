from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from .results import SMatrixData, LevelsData, BoundStateData
from .units import GHZ

def plot_scattering(ax: Axes, data: SMatrixData, plot_im_part: bool = True, x_units: float = 1., y_units: float = 1., **kwargs):
    color = kwargs.get("color")

    ax.plot(data.parameters() / x_units, data.s_length_re() / y_units, **kwargs)

    if plot_im_part:
        if color is None:
            color = ax.lines[-1].get_color()
            kwargs["color"] = color

        ax.plot(data.parameters() / x_units, data.s_length_im() / y_units, linestyle = "--", **kwargs)

def plot_levels(ax: Axes, data: LevelsData, x_units: float = 1., y_units: float = GHZ, **kwargs):
    ax.plot(data.parameters() / x_units, data.levels() / y_units, **kwargs)

# todo! add coloring for data with occupations
def plot_bound_states(ax: Axes, data: BoundStateData, x_units: float = 1., y_units: float = GHZ, **kwargs):
    for b in data:
        ax.plot(b.parameters() / x_units, b.bound_parameters() / y_units)