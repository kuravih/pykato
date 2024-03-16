from typing import Tuple
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


def Simple_Layout(left: float = 0.1, right: float = 0.1, bottom: float = 0.1, top: float = 0.1) -> Figure:
    """Layout a figure with an image.

        Example
    -------
        simple_layout_fig = Simple_Layout()
        ax, = simple_layout_fig.get_axes()

    Parameters
    ----------
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
    figure, (imshow_ax,) = GridSpec_Layout(gridspec.GridSpec(1, 1, left=left, bottom=bottom, right=1.0 - right, top=1.0 - top))

    return figure


def Image_Layout(left: float = 0.1, right: float = 0.1, bottom: float = 0.1, top: float = 0.1) -> Figure:
    """Layout a figure with an image.

    Example
    -------
        image_layout_fig = Image_Layout()
        imshow_ax, = image_layout_fig.get_axes()

    Parameters
    ----------
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

    figure, (imshow_ax,) = GridSpec_Layout(gridspec.GridSpec(1, 1, left=left, bottom=bottom, right=1.0 - right, top=1.0 - top))

    imshow_ax.set_aspect(1)

    return figure


def Image_Colorbar_Layout(width_ratios: Tuple[float, float] = (1, 0.05), wspace: float = 0.05, left: float = 0.1, right: float = 0.1, bottom: float = 0.1, top: float = 0.1) -> Figure:
    """Layout a figure with an image and a colorbar.

    Example
    -------
        image_colorbar_fig = Image_Colorbar_Layout()
        plot_ax, colorbar_ax = image_colorbar_fig.get_axes()

        image_colorbar_fig.set_data(...)

    Parameters
    ----------
        width_ratios : (float,float) = (image_width, colorbar_width)
            Width ratios of the axis objects
        wspace: float = 0.05
            Gap between image and colorbar
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

    figure, (imshow_ax, colorbar_ax) = GridSpec_Layout(gridspec.GridSpec(1, 2, left=left, bottom=bottom, right=1.0 - right, top=1.0 - top, wspace=wspace, width_ratios=width_ratios))

    imshow_ax.set_aspect(1)

    def _on_resize(_event):
        imshow_ax_bbox = imshow_ax.get_position()
        colorbar_ax_bbox = colorbar_ax.get_position()
        colorbar_ax.set_position((imshow_ax_bbox.x1 + colorbar_ax_bbox.width, imshow_ax_bbox.y0, colorbar_ax_bbox.width, imshow_ax_bbox.height))

    figure.resize = _on_resize

    return figure


def Image_Colorbar_Colorbar_Layout(width_ratios: Tuple[float, float] = (1, 0.05, 0.05), wspace: float = 0.05, left: float = 0.1, right: float = 0.1, bottom: float = 0.1, top: float = 0.1) -> Figure:
    """Layout a figure with a image and two colorbars.

    Example
    -------
        image_colorbar_colorbar_fig = Image_Colorbar_Colorbar_Layout()
        plot_ax, colorbar1_ax, colorbar2_ax = image_colorbar_fig.get_axes()

        image_colorbar_fig.set_data(...)

    Parameters
    ----------
        width_ratios : (float,float,float) = (image_width, colorbar_width, colorbar_width)
            Width ratios of the axis objects
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

    figure, (imshow_ax, colorbar1_ax, colorbar2_ax) = GridSpec_Layout(gridspec.GridSpec(nrows=1, ncols=3, left=left, bottom=bottom, right=1.0 - right, top=1.0 - top, wspace=wspace, width_ratios=width_ratios))

    imshow_ax.set_aspect(1)

    def _on_resize(_event):
        imshow_ax_bbox = imshow_ax.get_position()
        colorbar1_ax_bbox = colorbar1_ax.get_position()
        colorbar1_ax.set_position((imshow_ax_bbox.x1 + colorbar1_ax_bbox.width, imshow_ax_bbox.y0, colorbar1_ax_bbox.width, imshow_ax_bbox.height))
        colorbar2_ax.set_position((colorbar1_ax_bbox.x1 + colorbar1_ax_bbox.width, imshow_ax_bbox.y0, colorbar1_ax_bbox.width, imshow_ax_bbox.height))

    figure.resize = _on_resize

    return figure
