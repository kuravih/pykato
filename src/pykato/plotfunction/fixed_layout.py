from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Size, Divider
from matplotlib.figure import Figure


def Fixed_Layout(height: Tuple[float, ...] = (480.0,), width: Tuple[float, ...] = (480.0,), gap: Tuple[float, ...] = (21.0,), margin: Tuple[float, float, float, float] = (80.0, 60.0, 80.0, 70.0), scale: float = 0.0075) -> Figure:
    margin_l, margin_r, margin_b, margin_t = (amargin * scale for amargin in margin)

    assert len(gap) == len(width) - 1, f"number of gaps ({len(gap)}) must be 1 less than the number of widths ({len(width)})"

    axes_h = (aheight * scale for aheight in height)
    axes_w = (awidth * scale for awidth in width)
    axes_w_last = (width[0] if (len(width) == 1) else width[-1]) * scale
    gap_w = (agap * scale for agap in gap)

    grid_w = tuple([margin_l, *list(val for pair in zip(axes_w, gap_w) for val in pair), axes_w_last, margin_r])
    grid_h = (margin_t, *axes_h, margin_b)

    fig_w = np.sum(np.array(grid_w))
    fig_h = np.sum(np.array(grid_h))

    fig = plt.figure(figsize=(fig_w, fig_h))

    rect = (0.0, 0.0, 1.0, 1.0)
    xh = [Size.Fixed(_w) for _w in grid_w]
    yv = [Size.Fixed(_h) for _h in grid_h]
    divider = Divider(fig, rect, xh, yv, aspect=False)
    # -----------------------------------------------------------------------------------------------------------------

    for index, _ in enumerate(grid_w):
        if index % 2:
            _ax = fig.add_axes(rect)
            _ax.set_axes_locator(divider.new_locator(nx=index, ny=1))

    return fig


def Plot_Layout(size: Tuple[float, float] = (640.0, 480.0), margin: Tuple[float, float, float, float] = (100.0, 60.0, 60.0, 60.0), scale: float = 0.0075) -> Figure:
    """Layout a figure with a single axis.

    Example
    -------
        plot_fig = Plot_Layout()
        plot_ax, = plot_fig.get_axes()

        x = np.arange(100)
        y = np.random.randn(100)
        plot_ax.plot(x, y)

    Parameters
    ----------
        size : (float,float) = (width, height)
            Size of the axis object
        margin : (float,float,float,float) = (left, right, bottom, top)
            Margins.
        scale : float = scale
            A scale factor to multiply all measures. Used to get proper text sizing.

    Returns
    -------
        fig : plt.Figure
            Figure object.
    """
    return Fixed_Layout(height=(size[1],), width=(size[0],), gap=(), margin=margin, scale=scale)


def Image_Layout(size: Tuple[float, float] = (480.0, 480.0), margin: Tuple[float, float, float, float] = (80.0, 60.0, 80.0, 70.0), scale: float = 0.0075) -> Figure:
    """Layout a figure with a single axis.

    Example
    -------
        image_fig = Image_Layout()
        image_ax, = image_fig.get_axes()

    Parameters
    ----------
        size : (float,float) = (width, height)
            Size of the axis object
        margin : (float,float,float,float) = (left, right, bottom, top)
            Margins.
        scale : float = scale
            A scale factor to multiply all measures. Used to get proper text sizing.

    Returns
    -------
        fig : plt.Figure
            Figure object.
    """
    return Plot_Layout(size, margin, scale)


def Image_Colorbar_Layout(size: Tuple[float, float] = (480.0, 480.0), gap: float = 21.0, colorbar: float = 21.0, margin: Tuple[float, float, float, float] = (80.0, 60.0, 80.0, 70.0), scale: float = 0.0075) -> Figure:
    """Layout a figure with a image and a colorbar.

    Example
    -------
        image_colorbar_fig = Image_Colorbar_Layout()
        plot_ax, colorbar_ax = image_colorbar_fig.get_axes()

        image_colorbar_fig.set_data(...)

    Parameters
    ----------
        size : (float,float) = (width, height)
            Size of the axis object
        margin : (float,float,float,float) = (left, right, bottom, top)
            Margins.

        scale : float = scale
            A scale factor to multiply all measures. Used to get proper text sizing.

    Returns
    -------
        fig : plt.Figure
            Figure object.
    """
    return Fixed_Layout(height=(size[1],), width=(size[0], colorbar), gap=(gap,), margin=margin, scale=scale)


def Image_Colorbar_Colorbar_Layout(size: Tuple[float, float] = (480.0, 480.0), gap: Tuple[float, float] = (21.0, 42.0), colorbar: float = 21.0, margin: Tuple[float, float, float, float] = (80.0, 60.0, 80.0, 70.0), scale: float = 0.0075) -> Figure:
    return Fixed_Layout(height=(size[1],), width=(size[0], colorbar, colorbar), gap=gap, margin=margin, scale=scale)


def Image_Colorbar_Image_Colorbar_Layout(size: Tuple[float, float] = (480.0, 480.0), gap: Tuple[float, float, float] = (21.0, 210, 0), colorbar: float = 21.0, margin: Tuple[float, float, float, float] = (80.0, 60.0, 80.0, 70.0), scale: float = 0.0075) -> Figure:
    return Fixed_Layout(height=(size[1],), width=(size[0], colorbar, size[0], colorbar), gap=(gap[0], gap[1], gap[0]), margin=margin, scale=scale)


def Image_Image_Image_Colorbar_Colorbar_Layout(size: Tuple[float, float] = (480.0, 480.0), gap: Tuple[float, float, float] = (21.0, 210, 0), colorbar: float = 21.0, margin: Tuple[float, float, float, float] = (80.0, 60.0, 80.0, 70.0), scale: float = 0.0075) -> Figure:
    return Fixed_Layout(height=(size[1],), width=(size[0], size[0], size[0], colorbar, colorbar), gap=(gap[0], gap[0], gap[0], gap[1]), margin=margin, scale=scale)


def Image_Image_Image_Image_Colorbar_Colorbar_Layout(size: Tuple[float, float] = (480.0, 480.0), gap: Tuple[float, float, float] = (21.0, 210, 0), colorbar: float = 21.0, margin: Tuple[float, float, float, float] = (80.0, 60.0, 80.0, 70.0), scale: float = 0.0075) -> Figure:
    return Fixed_Layout(height=(size[1],), width=(size[0], size[0], size[0], size[0], colorbar, colorbar), gap=(gap[0], gap[0], gap[0], gap[0], gap[1]), margin=margin, scale=scale)


def Image_Image_Image_Image_Image_Colorbar_Colorbar_Layout(size: Tuple[float, float] = (480.0, 480.0), gap: Tuple[float, float, float] = (21.0, 210, 0), colorbar: float = 21.0, margin: Tuple[float, float, float, float] = (80.0, 60.0, 80.0, 70.0), scale: float = 0.0075) -> Figure:
    return Fixed_Layout(height=(size[1],), width=(size[0], size[0], size[0], size[0], size[0], colorbar, colorbar), gap=(gap[0], gap[0], gap[0], gap[0], gap[0], gap[1]), margin=margin, scale=scale)
