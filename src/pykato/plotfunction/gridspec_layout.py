from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def GridSpec_Layout(gridspecs) -> Tuple[Figure, Tuple[Axes, ...]]:
    figure = plt.figure()
    axs = ()
    for spec in gridspecs:
        axs = axs + (plt.subplot(spec),)
    return figure, axs


def Image_Colorbar_Layout(width_ratios: Tuple[float, float] = (10, 1), gap: float = 0.1, left: float = 0.1, right: float = 0.1, bottom: float = 0.1, top: float = 0.1) -> Figure:
    """Layout a figure with a image and a colorbar.

    Example
    -------
        image_colorbar_fig = Image_Colorbar_Layout()
        plot_ax, colorbar_ax = image_colorbar_fig.get_axes()

        image_colorbar_fig.set_data(...)

    Parameters
    ----------
        width_ratios : (float,float) = (image_width, colorbar_width)
            Width rations of the axis objects
        gap: float = 0.1
            WGap between image and colorbar
        left: float = 0.1
            Left Margin
        right: float = 0.1
            Right Margin
        bottom: float = 0.1
            Bottom Margin
        top: float = 0.1
            Top Margin

    Returns
    -------
        fig : plt.Figure
            Figure object.
    """

    figure, (imshow_ax, colorbar_ax) = GridSpec_Layout(gridspec.GridSpec(nrows=1, ncols=2, width_ratios=width_ratios))

    imshow_ax.set_aspect(1)

    def _on_resize(event):
        bbox = imshow_ax.get_position()
        _gap = bbox.width * gap
        _colorbar_width = bbox.width * (width_ratios[1] / width_ratios[0])
        colorbar_ax.set_position((bbox.x1 + _gap, bbox.y0, _colorbar_width, bbox.height))

    figure.resize = _on_resize

    return figure


def Image_Colorbar_Colorbar_Layout(width_ratios: Tuple[float, float, float] = (10, 1, 1), gap: float = 0.1, left: float = 0.1, right: float = 0.1, bottom: float = 0.1, top: float = 0.1) -> Figure:
    """Layout a figure with a image and two colorbars.

    Example
    -------
        image_colorbar_colorbar_fig = Image_Colorbar_Colorbar_Layout()
        plot_ax, colorbar1_ax, colorbar2_ax = image_colorbar_fig.get_axes()

        image_colorbar_fig.set_data(...)

    Parameters
    ----------
        width_ratios : (float,float,float) = (image_width, colorbar_width, colorbar_width)
            Width rations of the axis objects
        gap: float = 0.1
            WGap between image and colorbar
        left: float = 0.1
            Left Margin
        right: float = 0.1
            Right Margin
        bottom: float = 0.1
            Bottom Margin
        top: float = 0.1
            Top Margin

    Returns
    -------
        fig : plt.Figure
            Figure object.
    """

    figure, (imshow_ax, colorbar1_ax, colorbar2_ax) = GridSpec_Layout(gridspec.GridSpec(nrows=1, ncols=3, width_ratios=width_ratios))

    imshow_ax.set_aspect(1)

    def _on_resize(event):
        bbox = imshow_ax.get_position()
        _gap = bbox.width * gap
        _colorbar1_width = bbox.width * (width_ratios[1] / width_ratios[0])
        colorbar1_ax.set_position((bbox.x1 + _gap, bbox.y0, _colorbar1_width, bbox.height))
        _colorbar2_width = bbox.width * (width_ratios[2] / width_ratios[0])
        colorbar2_ax.set_position((bbox.x1 + _gap + _colorbar1_width + _gap, bbox.y0, _colorbar2_width, bbox.height))

    figure.resize = _on_resize

    return figure
