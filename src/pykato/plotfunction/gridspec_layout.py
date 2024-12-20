from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def _gridSpec_layout(gridspecs) -> Tuple[Figure, Tuple[Axes, ...]]:
    figure = plt.figure()
    axs = ()
    for spec in gridspecs:
        axs = axs + (plt.subplot(spec),)
    return figure, axs


def GridSpec_Layout(*args, aspect_ratios: Optional[Tuple[float, ...]] = None, **kwargs) -> Figure:
    """
    Layout a figure with multiple axes with specified aspect ratios.

    Examples:
        simple_layout_figure = GridSpec_Layout(1,1)
        image_layout_figure = GridSpec_Layout(1,1, aspect_ratios=(1,))
        fourimages_layout_figure = GridSpec_Layout(1,4, aspect_ratios=(1,1,1,1), width_ratios=(1,1,1,1), wspace=0.05)
        image_colorbar_layout_figure = GridSpec_Layout(1, 2, aspect_ratios=(1,0.05), width_ratios=(1,0.05), wspace=0.05)
        fourimages_colorbar_layout_figure = GridSpec_Layout(1, 5, aspect_ratios=(1,1,1,1,0.05), width_ratios=(1,1,1,1,0.05), wspace=0.05)
        image_twocolorbars_layout_figure = GridSpec_Layout(1, 3, aspect_ratios=(1,0.05,0.05), width_ratios=(1,0.05,0.05), wspace=0.05)
        fourimages_twocolorbars_layout_figure = GridSpec_Layout(1, 6, aspect_ratios=(1,1,1,1,0.05,0.05), width_ratios=(1,1,1,1,0.05,0.05), wspace=0.05)


    Parameters:
        aspect_ratios: Optional[Tuple[float, ...]]
            Aspect ratios for the exes.
        Other: matplotlib.gridspec.GridSpec parameters
            https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html

    Returns: plt.Figure
        Figure object.
    """

    figure, imshow_axes = _gridSpec_layout(gridspec.GridSpec(*args, **kwargs))

    if aspect_ratios is not None:
        for imshow_axis, aspect_ratio in zip(imshow_axes, aspect_ratios):
            imshow_axis.set_aspect(1.0 / aspect_ratio)

    return figure
