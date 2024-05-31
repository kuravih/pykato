from typing import Tuple, Optional, List
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.axis import Axis
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def Imshow_Preset(data: np.ndarray, figure: Optional[Figure] = None) -> Figure:
    """Preset with a single Imshow axis

    Examples
    --------
        figure = preset.Imshow_Preset(data)

    Parameters
    ----------
        data: np.ndarray
            Image data.
        figure: plt.Figure
            Figure object.

    Returns
    -------
        figure : plt.Figure
            Figure object with the imshow axis.

        Functions
        ---------
        figure.get_image() : AxesImage
            Get imshow image.

        figure.get_imshow_ax() : Axis
            Get imshow axis.

    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    imshow_image = imshow_ax.imshow(data)
    imshow_ax.invert_yaxis()

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_ax() -> Axis:
        return imshow_ax

    figure.get_imshow_ax = _get_imshow_ax
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def _complex_to_plot_arg_mod(zdata):
    arg_image = np.angle(zdata) / np.pi
    mod = np.abs(zdata)
    mod = (mod - np.nanmin(mod)) / (np.nanmax(mod) - np.nanmin(mod))
    mod_image = np.nan_to_num(mod, nan=0.0, posinf=1.0, neginf=0.0)
    return arg_image, mod_image


def Complex_Imshow_Preset(zdata: np.ndarray, figure: Optional[Figure] = None) -> Figure:
    """Preset with a single Imshow axis for complex fields.

    Examples
    --------
        figure = preset.Complex_Imshow_Preset(zdata)

    Parameters
    ----------
        zdata: np.ndarray
            Complex field data.
        figure: plt.Figure
            Figure object.

    Returns
    -------
        figure : plt.Figure
            Figure object with the imshow axis.

        Functions
        ---------
        figure.get_image() : AxesImage
            Get imshow image.

        figure.get_imshow_ax() : Axis
            Get imshow axis.

    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()

    arg_image, mod_image = _complex_to_plot_arg_mod(zdata)

    # approach 1: using a black/white background image without alpha + an hsv data image with alpha
    _bg_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)

    # # approach 2: using an hsv data background image without alpha + a black/white foreground image with alpha
    # imshow_image = imshow_ax.imshow(arg_image, cmap="hsv", vmin=-1, vmax=1)
    # _fg_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 1, dtype=float), "gray", alpha=1-mod_image, vmin=0, vmax=1)

    # TODO: out of the 2 approaches one is more convenient for using with set_data(complex) and set_alpha(normalized_intensity)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_ax() -> Axis:
        return imshow_ax

    figure.get_imshow_ax = _get_imshow_ax
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Preset(data: np.ndarray, figure: Optional[Figure] = None) -> Figure:
    """Preset with an Imshow axis and a colorbar axis.

    Examples
    --------
        figure = preset.Imshow_Colorbar_Preset(data)

    Parameters
    ----------
        data: np.ndarray
            Image data.
        figure: plt.Figure
            Figure object.

    Returns
    -------
        figure : plt.Figure
            Figure object with the imshow axis.

        Functions
        ---------
        figure.get_image() : AxesImage
            Get imshow image.

        figure.get_imshow_ax() : Axis
            Get imshow axis.

        figure.get_cbar_ax() : Axis
            Get colorbar axis.

    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    imshow_image = imshow_ax.imshow(data)
    imshow_ax.invert_yaxis()
    divider = make_axes_locatable(imshow_ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=colorbar_ax)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_ax() -> Axis:
        return imshow_ax

    figure.get_imshow_ax = _get_imshow_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_ax() -> Axis:
        return colorbar_ax

    figure.get_cbar_ax = _get_cbar_ax
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def _piFormatter(val, pos) -> str:
    num, den = abs(val).as_integer_ratio()
    sign = "" if (val >= 0) else "-"
    num_str = "\\pi" if (num == 1) else f"{num}\\pi"
    if num == 0:
        return "0"
    elif den == 1:
        return f"${sign}{num_str}$"
    else:
        return f"${sign}\\frac{{ {num_str} }}{{ {den} }}$"


def Complex_Imshow_2Colorbars_Preset(zdata: np.ndarray, figure: Optional[Figure] = None) -> Figure:
    """Preset with an Imshow axis and a colorbar axis.

    Examples
    --------
        figure = preset.Complex_Imshow_2Colorbars_Preset(data)

    Parameters
    ----------
        zdata: np.ndarray
            Complex field data.
        figure: plt.Figure
            Figure object.

    Returns
    -------
        figure : plt.Figure
            Figure object with the imshow axis.

        Functions
        ---------
        figure.get_image() : AxesImage
            Get imshow image.

        figure.get_image().set_dat(zdata)
            Get imshow image.

        figure.get_image().set_extent((xmin, xmax, ymin, ymax))
            Set image extent.

        figure.get_imshow_ax() : Axis
            Get imshow axis.

    """

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()

    arg_image, mod_image = _complex_to_plot_arg_mod(zdata)

    _bg_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)
    imshow_image.original_set_data = imshow_image.set_data
    imshow_image.set_data = None
    imshow_image.original_set_extent = imshow_image.set_extent
    imshow_image.set_extent = None
    divider = make_axes_locatable(imshow_ax)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_piFormatter))
    arg_colorbar_ax.set_title("arg", size=10)

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.4)
    figure.colorbar(_bg_imshow_image, cax=mod_colorbar_ax)
    mod_colorbar_ax.set_title("mod", size=10)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(zdata: np.ndarray) -> None:
        arg_image, mod_image = _complex_to_plot_arg_mod(zdata)
        imshow_image.original_set_data(arg_image)
        imshow_image.set_alpha(mod_image)

    imshow_image.set_data = _set_data
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_extent(extent) -> None:
        imshow_image.original_set_extent(extent)
        _bg_imshow_image.set_extent(extent)

    imshow_image.set_extent = _set_extent
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_ax() -> Axis:
        return imshow_ax

    figure.get_imshow_ax = _get_imshow_ax
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def ImageGrid_Preset(datas: Tuple[np.ndarray], figure: Optional[Figure] = None) -> Figure:
    """Preset with a grid of imshow axes.

    Examples
    --------
        figure = preset.ImageGrid_Preset((data1, data2, data3))

    Parameters
    ----------
        datas: Tuple[np.ndarray]
            Images
        figure: plt.Figure
            Figure object.

    Returns
    -------
        figure : plt.Figure
            Figure object with the imshow axis.

    """

    if figure is None:
        figure = plt.figure()

    imshow_axes = ImageGrid(figure, 111, nrows_ncols=(1, len(datas)), axes_pad=0.1)

    for imshow_axis, data in zip(imshow_axes, datas):
        imshow_axis.imshow(data)

    return figure


def ImageGrid_Colorbar_Preset(datas: Tuple[np.ndarray], figure: Optional[Figure] = None) -> Figure:
    """Preset with a grid of imshow axes and a colorbar.

    Examples
    --------
        figure = preset.ImageGrid_Colorbar_Preset((data1, data2, data3))

    Parameters
    ----------
        datas: Tuple[np.ndarray]
            Images
        figure: plt.Figure
            Figure object.

    Returns
    -------
        figure : plt.Figure
            Figure object with the imshow axis.

    """

    if figure is None:
        figure = plt.figure()

    imshow_axes = ImageGrid(figure, 111, nrows_ncols=(1, len(datas)), axes_pad=0.1, share_all=True, label_mode="L", cbar_location="right", cbar_mode="single")

    for imshow_axis, data in zip(imshow_axes, datas):
        imshow_image = imshow_axis.imshow(data)

    imshow_axes.cbar_axes[0].colorbar(imshow_image)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_ax() -> Axis:
        return imshow_axes.cbar_axes[0]

    figure.get_cbar_ax = _get_cbar_ax
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Complex_ImageGrid_2Colorbars_Preset(zdatas: Tuple[np.ndarray], figure: Optional[Figure] = None) -> Figure:
    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    divider = make_axes_locatable(imshow_ax)

    imshow_axes = [imshow_ax]
    imshow_images = []
    for index, zdata in enumerate(zdatas):
        arg_image, mod_image = _complex_to_plot_arg_mod(zdata)

        if index != 0:
            imshow_ax = divider.append_axes("right", size="100%", pad=0.1, sharex=imshow_ax, sharey=imshow_ax)
            imshow_ax.invert_yaxis()
            imshow_ax.yaxis.set_tick_params(labelleft=False)
            imshow_axes.append(imshow_ax)

        _bg_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
        imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)
        imshow_ax.invert_yaxis()
        imshow_images.append(imshow_image)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_piFormatter))
    arg_colorbar_ax.set_title("arg", size=10)

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.4)
    figure.colorbar(_bg_imshow_image, cax=mod_colorbar_ax)
    mod_colorbar_ax.set_title("mod", size=10)

    # -----------------------------------------------------------------------------------------------------------------
    def _plot(yxs: Tuple[np.ndarray, ...], *args, **kwargs) -> None:
        for yx, imshow_axis in zip(yxs, imshow_axes):
            imshow_axis.plot(yx[:, 1], yx[:, 0], *args, **kwargs)

    figure.plot = _plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(zdatas: Tuple[np.ndarray, ...]) -> None:
        for zdata, imshow_image in zip(zdatas, imshow_images):
            arg_image, mod_image = _complex_to_plot_arg_mod(zdata)
            imshow_image.set_data(arg_image)
            imshow_image.set_alpha(mod_image)

    figure.set_data = _set_data
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
    def _get_imshow_axes() -> List[Axis]:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    return figure
