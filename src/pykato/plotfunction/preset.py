from typing import Tuple, Optional, List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.axis import Axis
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from numpy.typing import NDArray


def Imshow_Preset(image: NDArray[np.float64], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with a single Imshow axis

    Examples:
        figure = Imshow_Preset(data)

    Parameters:
        image: NDArray[np.float64]
            Image data.

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
        Figure object with the imshow axis.

    Functions:
        get_imshow_ax(): Axes
            Get imshow axis.

        get_image(): AxesImage
            Get imshow image.

        close():
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    imshow_image = imshow_ax.imshow(image)
    imshow_ax.invert_yaxis()

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_ax() -> Axes:
        return imshow_ax

    figure.get_imshow_ax = _get_imshow_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def _complex_to_plot_arg_mod(zdata: NDArray[np.complex64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    arg_image = np.angle(zdata) / np.pi
    mod = np.abs(zdata)
    mod = (mod - np.nanmin(mod)) / (np.nanmax(mod) - np.nanmin(mod))
    mod_image = np.nan_to_num(mod, nan=0.0, posinf=1.0, neginf=0.0)
    return arg_image, mod_image


def Complex_Imshow_Preset(zimage: NDArray[np.complex64], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with a single Imshow axis for complex fields.

    Examples:
        figure = Complex_Imshow_Preset(zdata)

    Parameters:
        zdata: NDArray[np.complex64]
            Complex field data.

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_imshow_ax(): Axes
            Get imshow axis.

        get_image(): AxesImage
            Get imshow image.

        get_image().set_data(NDArray[np.complex64])
            Set image data.

        close():
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()

    arg_image, mod_image = _complex_to_plot_arg_mod(zimage)

    # approach 1: using a black/white background image without alpha + an hsv data image with alpha
    _bg_imshow_image = imshow_ax.imshow(np.full(zimage.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)
    imshow_ax.invert_yaxis()
    imshow_image.original_set_data = imshow_image.set_data
    imshow_image.set_data = None

    # # approach 2: using an hsv data background image without alpha + a black/white foreground image with alpha
    # imshow_image = imshow_ax.imshow(arg_image, cmap="hsv", vmin=-1, vmax=1)
    # _fg_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 1, dtype=float), "gray", alpha=1-mod_image, vmin=0, vmax=1)

    # TODO: out of the 2 approaches one is more convenient for using with set_data(complex) and set_alpha(normalized_intensity)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_ax() -> Axes:
        return imshow_ax

    figure.get_imshow_ax = _get_imshow_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(zimage: NDArray[np.complex64]) -> None:
        arg_image, mod_image = _complex_to_plot_arg_mod(zimage)
        imshow_image.original_set_data(arg_image)
        imshow_image.set_alpha(mod_image)

    imshow_image.set_data = _set_data
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Preset(image: NDArray[np.float64], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with an Imshow axis and a colorbar axis.

    Examples:
        figure = Imshow_Colorbar_Preset(data)

    Parameters:
        data: NDArray[np.float64]
            Image data.

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_image(): AxesImage
            Get imshow image.

        get_imshow_ax(): Axes
            Get imshow axis.

        get_cbar_ax(): Axes
            Get colorbar axis.

        close():
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    imshow_image = imshow_ax.imshow(image)
    imshow_ax.invert_yaxis()
    divider = make_axes_locatable(imshow_ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=colorbar_ax)
    figure.set_clim = imshow_image.set_clim

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_ax() -> Axes:
        return imshow_ax

    figure.get_imshow_ax = _get_imshow_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_ax() -> Axes:
        return colorbar_ax

    figure.get_cbar_ax = _get_cbar_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def _pi_formatter(val: float, pos) -> str:
    num, den = abs(val).as_integer_ratio()
    sign = "" if (val >= 0) else "-"
    num_str = "\\pi" if (num == 1) else f"{num}\\pi"
    if num == 0:
        return "0"
    elif den == 1:
        return f"${sign}{num_str}$"
    else:
        return f"${sign}\\frac{{ {num_str} }}{{ {den} }}$"


def Complex_Imshow_TwoColorbars_Preset(zimage: NDArray[np.complex64], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with an Imshow axis and two colorbar axes.

    Examples:
        figure = Complex_Imshow_TwoColorbars_Preset(data)

    Parameters:
        zimage: NDArray[np.complex64]
            Complex field data.

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_image(): AxesImage
            Get imshow image.

        get_image().set_data(zimage: NDArray[np.complex64])
            Set complex field data.

        get_imshow_ax(): Axes
            Get imshow axis.

        close():
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()

    arg_image, mod_image = _complex_to_plot_arg_mod(zimage)

    bg_imshow_image = imshow_ax.imshow(np.full(zimage.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)
    imshow_ax.invert_yaxis()
    imshow_image.original_set_data = imshow_image.set_data
    imshow_image.set_data = None
    divider = make_axes_locatable(imshow_ax)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_pi_formatter))
    arg_colorbar_ax.set_title("arg", size=10)

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.4)
    figure.colorbar(bg_imshow_image, cax=mod_colorbar_ax)
    mod_colorbar_ax.set_title("mod", size=10)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(zimage: NDArray[np.complex64]) -> None:
        arg_image, mod_image = _complex_to_plot_arg_mod(zimage)
        imshow_image.original_set_data(arg_image)
        imshow_image.set_alpha(mod_image)

    imshow_image.set_data = _set_data
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_ax() -> Axes:
        return imshow_ax

    figure.get_imshow_ax = _get_imshow_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def ImageGrid_Preset(images: List[NDArray[np.float64]], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with a grid of imshow axes.

    Examples:
        figure = ImageGrid_Preset([data1, data2, data3])

    Parameters:
        images: List[NDArray[np.float64]]
            List of images

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_images(): List[AxesImage]
            Get imshow image.

        get_images()[index].set_data(image: NDArray[np.float64])
            Set image data.

        close():
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    divider = make_axes_locatable(imshow_ax)

    imshow_axes = [imshow_ax]
    imshow_images = []
    for index, image in enumerate(images):
        if index != 0:
            imshow_ax = divider.append_axes("right", size="100%", pad=0.1, sharex=imshow_ax, sharey=imshow_ax)
            imshow_ax.invert_yaxis()
            imshow_ax.yaxis.set_tick_params(labelleft=False)
            imshow_axes.append(imshow_ax)

        imshow_image = imshow_ax.imshow(image)
        imshow_ax.invert_yaxis()
        imshow_images.append(imshow_image)

    # -----------------------------------------------------------------------------------------------------------------
    def _plot(xys: List[Tuple[np.float64, np.float64]], *args, **kwargs) -> None:
        for (x, y), imshow_axis in zip(xys, imshow_axes):
            imshow_axis.plot(x, y, *args, **kwargs)

    figure.plot = _plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_images() -> List[AxesImage]:
        return imshow_images

    figure.get_images = _get_images
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_xlim(*args, **kwargs)

    figure.set_xlim = _set_xlim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_ylim(*args, **kwargs)

    figure.set_ylim = _set_ylim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_xlabel(*args, **kwargs)

    figure.set_xlabel = _set_xlabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_ylabel(*args, **kwargs)

    figure.set_ylabel = _set_ylabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> List[Axes]:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def ImageGrid_Colorbar_Preset(images: List[NDArray[np.float64]], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with a grid of imshow axes and a colorbar.

    Examples:
        figure = preset.ImageGrid_Colorbar_Preset((data1, data2, data3))

    Parameters:
        data: List[np.ndarray]
            Images

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        plot(yxs: Tuple[np.ndarray, ...], *args, **kwargs)
            Plot on image grid

        get_images(): List[AxesImage]
            Get imshow images.

        set_clim(*args, **kwargs)
            Set clim.

        set_xlim(*args, **kwargs)
            Set xlim.

        set_ylim(*args, **kwargs)
            Set ylim.

        set_xlabel(*args, **kwargs)
            Set xlabel.

        set_ylabel(*args, **kwargs)
            Set ylabel.

        get_imshow_axes(): List[Axes]
            Get imshow exex

        get_cbar_ax(): Axis
            Get Colorbar axis

        close():
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    divider = make_axes_locatable(imshow_ax)

    imshow_axes = [imshow_ax]
    imshow_images = []
    for index, image in enumerate(images):
        if index != 0:
            imshow_ax = divider.append_axes("right", size="100%", pad=0.1, sharex=imshow_ax, sharey=imshow_ax)
            imshow_ax.invert_yaxis()
            imshow_ax.yaxis.set_tick_params(labelleft=False)
            imshow_axes.append(imshow_ax)

        imshow_image = imshow_ax.imshow(image)
        imshow_ax.invert_yaxis()
        imshow_images.append(imshow_image)

    colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=colorbar_ax)

    # -----------------------------------------------------------------------------------------------------------------
    def _plot(xys: List[Tuple[np.float64, np.float64]], *args, **kwargs) -> None:
        for (x, y), imshow_axis in zip(xys, imshow_axes):
            imshow_axis.plot(x, y, *args, **kwargs)

    figure.plot = _plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_images() -> List[AxesImage]:
        return imshow_images

    figure.get_images = _get_images
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_xlim(*args, **kwargs)

    figure.set_xlim = _set_xlim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_ylim(*args, **kwargs)

    figure.set_ylim = _set_ylim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_xlabel(*args, **kwargs)

    figure.set_xlabel = _set_xlabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_ylabel(*args, **kwargs)

    figure.set_ylabel = _set_ylabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> List[Axes]:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Complex_ImageGrid_TwoColorbars_Preset(zimage: List[NDArray[np.complex64]], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with grid of Imshow axes and two colorbar axes.

    Examples:
        figure = preset.Complex_Imshow_TwoColorbars_Preset(data)

    Parameters:
        zimage: List[NDArray[np.complex64]]
            Complex field data.

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        plot(yxs: Tuple[np.ndarray, ...], *args, **kwargs)
            Plot on image grid

        get_image().set_data(zdata)
            Get imshow image.

        get_image().set_extent((xmin, xmax, ymin, ymax))
            Set image extent.

        get_imshow_ax(): Axis
            Get imshow axis.

        close()
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    divider = make_axes_locatable(imshow_ax)

    imshow_axes = [imshow_ax]
    imshow_images = []
    for index, azdata in enumerate(zimage):
        arg_image, mod_image = _complex_to_plot_arg_mod(azdata)

        if index != 0:
            imshow_ax = divider.append_axes("right", size="100%", pad=0.1, sharex=imshow_ax, sharey=imshow_ax)
            imshow_ax.invert_yaxis()
            imshow_ax.yaxis.set_tick_params(labelleft=False)
            imshow_axes.append(imshow_ax)

        _bg_imshow_image = imshow_ax.imshow(np.full(azdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
        imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)
        imshow_image.original_set_data = imshow_image.set_data
        imshow_image.set_data = None
        imshow_ax.invert_yaxis()
        imshow_images.append(imshow_image)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_pi_formatter))
    arg_colorbar_ax.set_title("arg", size=10)

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.4)
    figure.colorbar(_bg_imshow_image, cax=mod_colorbar_ax)
    mod_colorbar_ax.set_title("mod", size=10)

    # -----------------------------------------------------------------------------------------------------------------
    plots = []
    def _plot(xys: List[Tuple[np.float64, np.float64]], *args, **kwargs) -> None:
        for (x, y), imshow_axis in zip(xys, imshow_axes):
            plot, = imshow_axis.plot(x, y, *args, **kwargs)
            plots.append(plot)

    figure.plot = _plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_plots():
        return plots

    figure.get_plots = _get_plots
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def __set_data(index: int, zimage: NDArray[np.complex64]) -> None:
        arg_image, mod_image = _complex_to_plot_arg_mod(zimage)
        imshow_images[index].original_set_data(arg_image)
        imshow_images[index].set_alpha(mod_image)

    for index, imshow_image in enumerate(imshow_images):
        imshow_image.set_data = lambda zimage, index=index: __set_data(index, zimage)
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_images() -> List[AxesImage]:
        return imshow_images

    figure.get_images = _get_images
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_xlim(*args, **kwargs)

    figure.set_xlim = _set_xlim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_ylim(*args, **kwargs)

    figure.set_ylim = _set_ylim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_xlabel(*args, **kwargs)

    figure.set_xlabel = _set_xlabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes:
            imshow_ax.set_ylabel(*args, **kwargs)

    figure.set_ylabel = _set_ylabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> List[Axes]:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Imshow_Colorbar_Preset(images: Tuple[NDArray[np.float64], NDArray[np.float64]], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with two Imshow axis and Colorbar axis pairs.

    Examples:
        figure = preset.Imshow_Colorbar_Imshow_Colorbar_Preset((imageA, imageB))

    Parameters:
        data: Tuple[NDArray[np.float64], NDArray[np.float64]]
            Tuple of two images.

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_images(): List[AxesImage]
            Get imshow image.

        get_images()[index].set_data(image:NDArray[np.float64])
            Get imshow image.

        get_imshow_axes(): List[Axis]
            Get imshow axis.

        close()
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    image_a, image_b = images

    imshow_ax_a = figure.gca()
    imshow_image_a = imshow_ax_a.imshow(image_a)
    imshow_ax_a.invert_yaxis()

    divider = make_axes_locatable(imshow_ax_a)

    colorbar_ax_a = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image_a, cax=colorbar_ax_a)

    imshow_ax_b = divider.append_axes("right", size="100%", pad=1.0)
    imshow_image_b = imshow_ax_b.imshow(image_b)
    imshow_ax_b.invert_yaxis()

    colorbar_ax_b = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image_b, cax=colorbar_ax_b)

    imshow_axes = [imshow_ax_a, imshow_ax_b]
    imshow_images = [imshow_image_a, imshow_image_b]
    colorbar_axes = [colorbar_ax_a, colorbar_ax_b]

    # -----------------------------------------------------------------------------------------------------------------
    def _get_images() -> List[AxesImage]:
        return imshow_images

    figure.get_images = _get_images
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> List[Axis]:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes() -> Axes:
        return colorbar_axes

    figure.get_cbar_axes = _get_cbar_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Histogram_Colorbar_Preset(data: np.ndarray, figure: Optional[Figure] = None, nbins=256, vmin: float = 0, vmax: float = 1.0, cmap: str = "viridis", position: str = "right") -> Figure:
    """
    Preset with an Simple axis and a colorbar axis.
    Suitable for visualizing histograms of color data.

    Examples:
        figure = preset.Simple_Colorbar_Preset(data)

    Parameters:
        data: np.ndarray
            Image data.

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        set_data(data: np.ndarray)
            Set histogram data.

        set_vlim(vmin:float, vmax:float)
            Set minimum and maximum values.

        get_histogram_ax(): Axes
            Get histogram axis.

        get_cbar_ax(): Axes
            Get colorbar axis.

        close()
            Properly close the figure.

    """

    if figure is None:
        figure = plt.figure()

    figure.data = data
    figure.vmin, figure.vmax = vmin, vmax
    figure.nbins = nbins

    colormap = colormaps[cmap]

    normalize_fn = Normalize(vmin=figure.vmin, vmax=figure.vmax)
    bar_xs, bar_width = np.linspace(figure.vmin, figure.vmax, figure.nbins + 1, retstep=True)
    figure.bar_xcs = bar_xs + bar_width / 2
    bar_colors = colormap(normalize_fn(figure.bar_xcs))

    histogram_ax = figure.gca()
    divider = make_axes_locatable(histogram_ax)
    colorbar_ax = divider.append_axes(position, size="5%", pad=0.1, sharex=histogram_ax)
    ColorbarBase(colorbar_ax, cmap=colormap, norm=normalize_fn, orientation="horizontal")

    bar_hs, _ = np.histogram(figure.data, figure.bar_xcs)

    hist_bars = histogram_ax.bar(figure.bar_xcs[:-1], bar_hs, width=bar_width, color=bar_colors)
    histogram_ax.set_xlim(figure.vmin, figure.vmax)
    histogram_ax.set_xlabel(None)
    histogram_ax.tick_params(axis="x", which="both", labelbottom=False)

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(data: np.ndarray):
        figure.data = data
        bar_hs, _ = np.histogram(figure.data, figure.bar_xcs)
        for hist_bar, bar_h in zip(hist_bars, bar_hs):
            hist_bar.set_height(bar_h)

    figure.set_data = _set_data
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_histogram_ax() -> Axes:
        return histogram_ax

    figure.get_histogram_ax = _get_histogram_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_ax() -> Axis:
        return colorbar_ax

    figure.get_cbar_ax = _get_cbar_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_vlim(vmin: float, vmax: float):
        figure.vmin, figure.vmax = vmin, vmax
        normalize_fn = Normalize(vmin=figure.vmin, vmax=figure.vmax)
        ColorbarBase(colorbar_ax, cmap=colormap, norm=normalize_fn, orientation="horizontal")
        bar_xs, bar_width = np.linspace(figure.vmin, figure.vmax, figure.nbins + 1, retstep=True)
        figure.bar_xcs = bar_xs + bar_width / 2
        bar_colors = colormap(normalize_fn(figure.bar_xcs))
        bar_hs, _ = np.histogram(figure.data, figure.bar_xcs)
        for hist_bar, bar_h, bar_x, bar_color in zip(hist_bars, bar_hs, bar_xs, bar_colors):
            hist_bar.set_x(bar_x)
            hist_bar.set_width(bar_width)
            hist_bar.set_height(bar_h)
            hist_bar.set_color(bar_color)

    figure.set_vlim = _set_vlim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset(data: Tuple[NDArray[np.float64], NDArray[np.float64]], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with two Imshow axis and Colorbar axis pairs and two axis for plots

    Examples:
        figure = preset.Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset((imageA, imageB))

    Parameters:
        data: Tuple[NDArray[np.float64], NDArray[np.float64]]
            Tuple of two images.

        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_images(): List[AxesImage]
            Get imshow image.

        get_images()[index].set_data(image: NDArray[np.float64])
            Set imshow image.

        get_imshow_axes(): List[Axis]
            Get imshow axis.

        get_plot_axes(): List[Axis]
            Get plot axis.

        get_cbar_ax(): Axes
            Get colorbar axis.

        close()
            Properly close the figure.
    """

    data_a, data_b = data

    if figure is None:
        figure = plt.figure()

    gs1 = GridSpec(nrows=3, ncols=1, height_ratios=(1, 0.5, 0.5), hspace=0.5, figure=figure)

    plot_ax_1 = figure.add_subplot(gs1[1])

    plot_ax_2 = figure.add_subplot(gs1[2])

    gs2 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs1[0])

    imshow_ax_a = figure.add_subplot(gs2[0])
    imshow_image_a = imshow_ax_a.imshow(data_a)
    imshow_ax_a.invert_yaxis()

    divider_a = make_axes_locatable(imshow_ax_a)

    colorbar_ax_a = divider_a.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image_a, cax=colorbar_ax_a)

    imshow_ax_b = figure.add_subplot(gs2[1])
    imshow_image_b = imshow_ax_b.imshow(data_b)
    imshow_ax_b.invert_yaxis()

    divider_b = make_axes_locatable(imshow_ax_b)

    colorbar_ax_b = divider_b.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image_b, cax=colorbar_ax_b)

    plot_axes = [plot_ax_1, plot_ax_2]
    imshow_axes = [imshow_ax_a, imshow_ax_b]
    imshow_images = [imshow_image_a, imshow_image_b]
    colorbar_axes = [colorbar_ax_a, colorbar_ax_b]

    # -----------------------------------------------------------------------------------------------------------------
    def _get_images() -> List[AxesImage]:
        return imshow_images

    figure.get_images = _get_images
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> List[Axes]:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_plot_axes() -> List[Axes]:
        return plot_axes

    figure.get_plot_axes = _get_plot_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes() -> List[Axis]:
        return colorbar_axes

    figure.get_cbar_axes = _get_cbar_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Imshow_Colorbar_Plot_Preset(data: Tuple[NDArray[np.float64], NDArray[np.float64]], figure: Optional[Figure] = None) -> Figure:
    """
    Preset with two Imshow axis and Colorbar axis pairs and two axis for plots

    Examples:
        figure = preset.Imshow_Colorbar_Imshow_Colorbar_Plot_Preset((imageA, imageB))

    Parameters:
        data: Tuple[np.ndarray, np.ndarray]
            Tuple of two images.
        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_images(): List[AxesImage]
            Get imshow image.

        get_images()[index].set_data(image: NDArray[np.float64])
            Set imshow image.

        get_imshow_axes(): List[Axis]
            Get imshow axis.

        get_plot_axes(): Axis
            Get plot axis.

        get_cbar_ax(): Axes
            Get colorbar axis.

        close()
            Properly close the figure.
    """

    data_a, data_b = data

    if figure is None:
        figure = plt.figure()

    gs1 = GridSpec(nrows=2, ncols=1, height_ratios=(1, 0.5), hspace=0.25, figure=figure)

    plot_ax = figure.add_subplot(gs1[1])

    gs2 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs1[0])

    imshow_ax_a = figure.add_subplot(gs2[0])
    imshow_image_a = imshow_ax_a.imshow(data_a)
    imshow_ax_a.invert_yaxis()

    divider_a = make_axes_locatable(imshow_ax_a)

    colorbar_ax_a = divider_a.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image_a, cax=colorbar_ax_a)

    imshow_ax_b = figure.add_subplot(gs2[1])
    imshow_image_b = imshow_ax_b.imshow(data_b)
    imshow_ax_b.invert_yaxis()

    divider_b = make_axes_locatable(imshow_ax_b)

    colorbar_ax_b = divider_b.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image_b, cax=colorbar_ax_b)

    imshow_axes = [imshow_ax_a, imshow_ax_b]
    imshow_images = [imshow_image_a, imshow_image_b]
    colorbar_axes = [colorbar_ax_a, colorbar_ax_b]

    # -----------------------------------------------------------------------------------------------------------------
    def _get_images() -> List[AxesImage]:
        return imshow_images

    figure.get_images = _get_images
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> List[Axes]:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_plot_ax() -> Axes:
        return plot_ax

    figure.get_plot_ax = _get_plot_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes() -> List[Axis]:
        return colorbar_axes

    figure.get_cbar_axes = _get_cbar_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure
