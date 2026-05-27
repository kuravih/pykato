import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.axis import Axis
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, LinearLocator
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fractions import Fraction
from matplotlib.typing import ColorType

import numpy as np
from numpy.typing import NDArray

from pykato.log import setup_logger


logger = setup_logger("preset", terminator="\n")

def monkeypatch_Axes_mask_image(axes: Axes, mask: NDArray[np.float64]):
    """
    monkey patch get_mask():NDArray[float] to an AxesImage (Remember to use axesImage.get_mask().set_data(...) or axesImage.get_mask().set_alpha(...) afterwards to update the image)
    """
    axes._mask_image = axes.imshow(np.zeros_like(mask), alpha=mask)

    def _get_mask() -> AxesImage:
        return axes._mask_image

    axes.get_mask = _get_mask


def monkeypatch_AxesImage_alpha_mask_show(axesImage: AxesImage, alpha_mask: NDArray[np.float64], color: ColorType):
    """
    monkey patch set_alpha_mask_show(value: bool), get_alpha_mask_show():bool to an AxesImage
    """
    axesImage._alpha_mask_show: bool = False

    def _get_alpha_mask_show() -> bool:
        return axesImage._alpha_mask_show

    axesImage.get_alpha_mask_show = _get_alpha_mask_show

    def _set_alpha_mask_show(value: bool):
        axesImage._alpha_mask_show = value
        if axesImage._alpha_mask_show & (alpha_mask is not None):
            axesImage.set_alpha(alpha_mask)
            axesImage.axes.set_facecolor(color)
        else:
            axesImage.set_alpha(None)

    axesImage.set_alpha_mask_show = _set_alpha_mask_show


def monkeypatch_AxesImage_cmap_name(axesImage: AxesImage):
    """
    monkey patch set_cmap_name(value: str), get_cmap_norm():str to an AxesImage
    """
    axesImage._cmap_name: str = "viridis"

    def _get_cmap_name() -> str:
        return axesImage._cmap_name

    axesImage.get_cmap_name = _get_cmap_name

    def _set_cmap_name(value: str, bad: str | None = "black", over: str | None = None):
        axesImage._cmap_name = value
        _cmap = mpl.colormaps[axesImage._cmap_name].copy()
        if bad is not None:
            _cmap.set_bad(color=bad)
        if over is not None:
            _cmap.set_over(over)
        axesImage.set_cmap(_cmap)

    axesImage.set_cmap_name = _set_cmap_name


def monkeypatch_AxesImage_cmap_norm(axesImage: AxesImage):
    """
    monkey patch set_cmap_norm(value: bool), get_cmap_norm():bool to an AxesImage
    """
    axesImage._cmap_norm: Normalize = Normalize()

    def _get_cmap_norm() -> Normalize:
        return axesImage._cmap_norm

    axesImage.get_cmap_norm = _get_cmap_norm

    def _set_cmap_norm(value: Normalize):
        axesImage._cmap_norm = value
        axesImage.set_norm(value)

    axesImage.set_cmap_norm = _set_cmap_norm


def Imshow_Preset(image: NDArray[np.float64], figure: Figure | None = None) -> Figure:
    """
    Preset with a single Imshow axis

    Examples:
        figure = Imshow_Preset(data)

    Parameters:
        image: NDArray[np.float64]
            Image data.

        figure: Figure | None = None
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

    imshow_axes = figure.gca()

    imshow_image = imshow_axes.imshow(image)
    imshow_axes.invert_yaxis()

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> Axes:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
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


def _complex_to_plot_abs_arg(zdata: NDArray[np.complex64], abs_min: float | None = None, abs_max: float | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    arg_image = np.angle(zdata)
    _abs = np.abs(zdata)
    if abs_min is None:
        abs_min = np.nanmin(_abs)
    if abs_max is None:
        abs_max = np.nanmax(_abs)
    _abs =  (_abs - abs_min) / (abs_max - abs_min)
    _abs = np.nan_to_num(_abs, nan=0.0, posinf=1.0, neginf=0.0)
    abs_image = np.clip(_abs, 0.0, 1.0)
    return abs_image, arg_image


def Complex_Imshow_Preset(zimage: NDArray[np.complex64], figure: Figure | None = None) -> Figure:
    """
    Preset with a single Imshow axis for complex fields.

    Examples:
        figure = Complex_Imshow_Preset(zdata)

    Parameters:
        zdata: NDArray[np.complex64]
            Complex field data.

        figure: Figure | None = None
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

    imshow_axes = figure.gca()

    abs_image, arg_image = _complex_to_plot_abs_arg(zimage)

    # approach 1: using a black/white background image without alpha + an hsv data image with alpha
    # _bg_imshow_image = imshow_axes.imshow(np.zeros_like(zimage, dtype=float), "gray", vmin=0, vmax=1)
    imshow_axes.set_facecolor("black") # or white
    imshow_image = imshow_axes.imshow(arg_image, alpha=abs_image, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    imshow_axes.invert_yaxis()
    imshow_image.original_set_data = imshow_image.set_data
    imshow_image.set_data = None

    # # approach 2: using an hsv data background image without alpha + a black/white foreground image with alpha
    # imshow_image = imshow_ax.imshow(arg_image, cmap="hsv", vmin=-1, vmax=1)
    # _fg_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 1, dtype=float), "gray", alpha=1-abs_image, vmin=0, vmax=1)

    # TODO: out of the 2 approaches one is more convenient for using with set_data(complex) and set_alpha(normalized_intensity)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> Axes:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(zimage: NDArray[np.complex64]) -> None:
        abs_image, arg_image = _complex_to_plot_abs_arg(zimage)
        imshow_image.original_set_data(arg_image)
        imshow_image.set_alpha(abs_image)

    imshow_image.set_data = _set_data
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Preset(image: NDArray[np.float64], figure: Figure | None = None) -> Figure:
    """
    Preset with an Imshow axis and a colorbar axis.

    Examples:
        figure = Imshow_Colorbar_Preset(data)

    Parameters:
        data: NDArray[np.float64]
            Image data.

        figure: Figure | None = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_cmap_name(): str
            Get colormap name.

        set_cmap_name(value: str, bad: str | None = 'black', over: str | None = None):
            Set colormap by name.

        get_image(): AxesImage
            Get imshow image.

        get_imshow_axes(): Axes
            Get imshow axes.

        get_cbar_axes(): Axes
            Get colorbar axes.

        close():
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    imshow_axes = figure.gca()

    imshow_image = imshow_axes.imshow(image)
    imshow_axes.invert_yaxis()

    divider = make_axes_locatable(imshow_axes)

    colorbar_axes = divider.append_axes("right", size="5%", pad=0.1)

    figure.colorbar(imshow_image, cax=colorbar_axes)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> Axes:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes() -> Axes:
        return colorbar_axes

    figure.get_cbar_axes = _get_cbar_axes
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


def _pi_formatter(val: float, pos) -> str:
    if abs(val) < 1e-10:  # Handle values very close to zero
        return "0"

    fraction_of_pi = Fraction(val / np.pi).limit_denominator()  # Convert to fraction of π

    sign = "" if fraction_of_pi >= 0 else "-"
    abs_frac = abs(fraction_of_pi)

    if abs_frac.numerator == 0:
        return "0"
    elif abs_frac.numerator == 1 and abs_frac.denominator == 1:
        return f"${sign}\\pi$"
    elif abs_frac.numerator == 1:
        return f"${sign}\\frac{{ \\pi }}{{ {abs_frac.denominator} }}$"
    elif abs_frac.denominator == 1:
        return f"${sign}{abs_frac.numerator}\\pi$"
    else:
        return f"${sign}\\frac{{ {abs_frac.numerator}\\pi }}{{ {abs_frac.denominator} }}$"


def Complex_Imshow_TwoColorbars_Preset(zimage: NDArray[np.complex128], abs_min: float | None = None, abs_max: float | None = None, figure: Figure | None = None) -> Figure:
    """
    Preset with an Imshow axis and two colorbar axes.

    Examples:
        figure = Complex_Imshow_TwoColorbars_Preset(data)

    Parameters:
        zimage: NDArray[np.complex64]
            Complex field data.

        figure: Figure | None = None
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

    imshow_axes = figure.gca()

    _abs = np.abs(zimage)
    if abs_min is None:
        abs_min = float(np.nanmin(_abs))
    if abs_max is None:
        abs_max = float(np.nanmax(_abs))

    abs_image, arg_image = _complex_to_plot_abs_arg(zimage, abs_min, abs_max)

    bg_imshow_image = imshow_axes.imshow(np.zeros_like(zimage, dtype=float), "gray", vmin=0, vmax=1)
    # imshow_axes.set_facecolor("black") # or white
    imshow_image = imshow_axes.imshow(arg_image, alpha=abs_image, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    imshow_axes.invert_yaxis()
    imshow_image.original_set_data = imshow_image.set_data
    imshow_image.set_data = None
    divider = make_axes_locatable(imshow_axes)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_locator(LinearLocator(numticks=9))
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_pi_formatter))
    arg_colorbar_ax.set_title("arg", size=10)

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.4)
    mod_sm = ScalarMappable(norm=Normalize(vmin=abs_min, vmax=abs_max), cmap="gray")
    mod_sm.set_array([])
    figure.colorbar(mod_sm, cax=mod_colorbar_ax)
    mod_colorbar_ax.set_title("mod", size=10)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes() -> Axes:
        return imshow_axes

    figure.get_imshow_axes = _get_imshow_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes_list() -> list[Axes]:
        return [arg_colorbar_ax, mod_colorbar_ax]

    figure.get_cbar_axes_list = _get_cbar_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(zimage: NDArray[np.complex64], abs_min: float | None = None, abs_max: float | None = None) -> None:
        abs_image, arg_image = _complex_to_plot_abs_arg(zimage, abs_min, abs_max)
        imshow_image.original_set_data(arg_image)
        imshow_image.set_alpha(abs_image)

    imshow_image.set_data = _set_data
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def ImageGrid_Preset(images: list[NDArray[np.float64]], figure: Figure | None = None) -> Figure:
    """
    Preset with a grid of imshow axes.

    Examples:
        figure = ImageGrid_Preset([data1, data2, data3])

    Parameters:
        images: list[NDArray[np.float64]]
            List of images

        figure: Figure | None = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_images(): list[AxesImage]
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

    imshow_axes_list = [imshow_ax]
    imshow_image_list = []
    for index, image in enumerate(images):
        if index != 0:
            imshow_ax = divider.append_axes("right", size="100%", pad=0.1, sharex=imshow_ax, sharey=imshow_ax)
            imshow_ax.invert_yaxis()
            imshow_ax.yaxis.set_tick_params(labelleft=False)
            imshow_axes_list.append(imshow_ax)

        imshow_image = imshow_ax.imshow(image)
        imshow_ax.invert_yaxis()
        imshow_image_list.append(imshow_image)

    # -----------------------------------------------------------------------------------------------------------------
    def _plot(xys: list[tuple[np.float64, np.float64]], *args, **kwargs) -> None:
        for (x, y), imshow_axis in zip(xys, imshow_axes_list):
            imshow_axis.plot(x, y, *args, **kwargs)

    figure.plot = _plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image_list() -> list[AxesImage]:
        return imshow_image_list

    figure.get_image_list = _get_image_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes_list:
            imshow_ax.set_xlim(*args, **kwargs)

    figure.set_xlim = _set_xlim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes_list:
            imshow_ax.set_ylim(*args, **kwargs)

    figure.set_ylim = _set_ylim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes_list:
            imshow_ax.set_xlabel(*args, **kwargs)

    figure.set_xlabel = _set_xlabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes_list:
            imshow_ax.set_ylabel(*args, **kwargs)

    figure.set_ylabel = _set_ylabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes_list() -> list[Axes]:
        return imshow_axes_list

    figure.get_imshow_axes_list = _get_imshow_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def ImageGrid_Colorbar_Preset(images: list[NDArray[np.float64]], figure: Figure | None = None) -> Figure:
    """
    Preset with a grid of imshow axes and a colorbar.

    Examples:
        figure = preset.ImageGrid_Colorbar_Preset((data1, data2, data3))

    Parameters:
        data: list[np.ndarray]
            Images

        figure: Figure | None = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        plot(yxs: tuple[np.ndarray, ...], *args, **kwargs)
            Plot on image grid

        get_images(): list[AxesImage]
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

        get_imshow_axes(): list[Axes]
            Get imshow exex

        get_cbar_ax(): Axis
            Get Colorbar axis

        close():
            Properly close the figure.
    """

    if figure is None:
        figure = plt.figure()

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")

    imshow_ax = figure.gca()
    divider = make_axes_locatable(imshow_ax)

    imshow_axes_list = [imshow_ax]
    imshow_image_list = []
    for index, image in enumerate(images):
        if index != 0:
            imshow_ax = divider.append_axes("right", size="100%", pad=0.1, sharex=imshow_ax, sharey=imshow_ax)
            imshow_ax.invert_yaxis()
            imshow_ax.yaxis.set_tick_params(labelleft=False)
            imshow_axes_list.append(imshow_ax)

        imshow_image = imshow_ax.imshow(image, cmap=cmap)
        imshow_ax.invert_yaxis()
        imshow_image_list.append(imshow_image)

    colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=colorbar_ax)

    # -----------------------------------------------------------------------------------------------------------------
    def _plot(xys: list[tuple[np.float64, np.float64]], *args, **kwargs) -> None:
        for (x, y), imshow_axis in zip(xys, imshow_axes_list):
            imshow_axis.plot(x, y, *args, **kwargs)

    figure.plot = _plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image_list() -> list[AxesImage]:
        return imshow_image_list

    figure.get_image_list = _get_image_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes_list:
            imshow_ax.set_xlim(*args, **kwargs)

    figure.set_xlim = _set_xlim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylim(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes_list:
            imshow_ax.set_ylim(*args, **kwargs)

    figure.set_ylim = _set_ylim
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_xlabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes_list:
            imshow_ax.set_xlabel(*args, **kwargs)

    figure.set_xlabel = _set_xlabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_ylabel(*args, **kwargs) -> None:
        for imshow_ax in imshow_axes_list:
            imshow_ax.set_ylabel(*args, **kwargs)

    figure.set_ylabel = _set_ylabel
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes_list() -> list[Axes]:
        return imshow_axes_list

    figure.get_imshow_axes_list = _get_imshow_axes_list
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


def Complex_ImageGrid_TwoColorbars_Preset(zimage: list[NDArray[np.complex64]], figure: Figure | None = None) -> Figure:
    """
    Preset with grid of Imshow axes and two colorbar axes.

    Examples:
        figure = preset.Complex_ImageGrid_TwoColorbars_Preset(data)

    Parameters:
        zimage: list[NDArray[np.complex64]]
            Complex field data.

        figure: Figure | None = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        plot(yxs: tuple[np.ndarray, ...], *args, **kwargs)
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

    imshow_axes = figure.gca()

    divider = make_axes_locatable(imshow_axes)

    imshow_axes_list = [imshow_axes]
    imshow_image_list = []
    for index, azdata in enumerate(zimage):
        abs_image, arg_image = _complex_to_plot_abs_arg(azdata)

        if index != 0:
            imshow_axes = divider.append_axes("right", size="100%", pad=0.1, sharex=imshow_axes, sharey=imshow_axes)
            imshow_axes.invert_yaxis()
            imshow_axes.yaxis.set_tick_params(labelleft=False)
            imshow_axes_list.append(imshow_axes)

        _bg_imshow_image = imshow_axes.imshow(np.full(azdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
        # imshow_axes.set_facecolor("black") # or white
        imshow_image = imshow_axes.imshow(arg_image, alpha=abs_image, cmap="hsv", vmin=-np.pi, vmax=np.pi)
        imshow_image.original_set_data = imshow_image.set_data
        imshow_image.set_data = None
        imshow_axes.invert_yaxis()
        imshow_image_list.append(imshow_image)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_locator(LinearLocator(numticks=9))
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_pi_formatter))
    arg_colorbar_ax.set_title("arg", size=10)

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.4)
    figure.colorbar(_bg_imshow_image, cax=mod_colorbar_ax)
    mod_colorbar_ax.set_title("mod", size=10)

    # -----------------------------------------------------------------------------------------------------------------
    def __set_data(index: int, zimage: NDArray[np.complex64]) -> None:
        abs_image, arg_image = _complex_to_plot_abs_arg(zimage)
        imshow_image_list[index].original_set_data(arg_image)
        imshow_image_list[index].set_alpha(abs_image)

    for index, imshow_image in enumerate(imshow_image_list):
        imshow_image.set_data = lambda zimage, index=index: __set_data(index, zimage)
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image_list() -> list[AxesImage]:
        return imshow_image_list

    figure.get_image_list = _get_image_list
    # -----------------------------------------------------------------------------------------------------------------

    # # -----------------------------------------------------------------------------------------------------------------
    # def _set_xlim(*args, **kwargs) -> None:
    #     for imshow_ax in imshow_axes_list:
    #         imshow_ax.set_xlim(*args, **kwargs)

    # figure.set_xlim = _set_xlim
    # # -----------------------------------------------------------------------------------------------------------------

    # # -----------------------------------------------------------------------------------------------------------------
    # def _set_ylim(*args, **kwargs) -> None:
    #     for imshow_ax in imshow_axes_list:
    #         imshow_ax.set_ylim(*args, **kwargs)

    # figure.set_ylim = _set_ylim
    # # -----------------------------------------------------------------------------------------------------------------

    # # -----------------------------------------------------------------------------------------------------------------
    # def _set_xlabel(*args, **kwargs) -> None:
    #     for imshow_ax in imshow_axes_list:
    #         imshow_ax.set_xlabel(*args, **kwargs)

    # figure.set_xlabel = _set_xlabel
    # # -----------------------------------------------------------------------------------------------------------------

    # # -----------------------------------------------------------------------------------------------------------------
    # def _set_ylabel(*args, **kwargs) -> None:
    #     for imshow_ax in imshow_axes_list:
    #         imshow_ax.set_ylabel(*args, **kwargs)

    # figure.set_ylabel = _set_ylabel
    # # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes_list() -> list[Axes]:
        return imshow_axes_list

    figure.get_imshow_axes_list = _get_imshow_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Imshow_Colorbar_Preset(images: tuple[NDArray[np.float64], NDArray[np.float64]], figure: Figure | None = None) -> Figure:
    """
    Preset with two Imshow axis and Colorbar axis pairs.

    Examples:
        figure = preset.Imshow_Colorbar_Imshow_Colorbar_Preset((imageA, imageB))

    Parameters:
        data: tuple[NDArray[np.float64], NDArray[np.float64]]
            Tuple of two images.

        figure: Figure | None = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_images(): tuple[AxesImage, AxesImage]:
            Get imshow image.

        get_images()[index].set_data(image:NDArray[np.float64])
            Get imshow image.

        get_imshow_axes(): list[Axis]
            Get imshow axis.

        close()
            Properly close the figure.
    """

    image_a, image_b = images

    if figure is None:
        figure = plt.figure()

    imshow_axes_a = figure.gca()

    imshow_image_a = imshow_axes_a.imshow(image_a)
    imshow_axes_a.invert_yaxis()

    divider = make_axes_locatable(imshow_axes_a)

    cbar_axes_a = divider.append_axes("right", size="5%", pad=0.1)

    figure.colorbar(imshow_image_a, cax=cbar_axes_a)

    imshow_axes_b = divider.append_axes("right", size="100%", pad=1.0)

    imshow_image_b = imshow_axes_b.imshow(image_b)
    imshow_axes_b.invert_yaxis()

    cbar_axes_b = divider.append_axes("right", size="5%", pad=0.1)

    figure.colorbar(imshow_image_b, cax=cbar_axes_b)

    imshow_image_list = [imshow_image_a, imshow_image_b]
    imshow_axes_list = [imshow_axes_a, imshow_axes_b]
    cbar_axes_list = [cbar_axes_a, cbar_axes_b]

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image_list() -> list[AxesImage]:
        return imshow_image_list

    figure.get_image_list = _get_image_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes_list() -> list[Axes]:
        return imshow_axes_list

    figure.get_imshow_axes_list = _get_imshow_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes_tuple() -> tuple[Axes, Axes]:
        return cbar_axes_list

    figure.get_cbar_axes_tuple = _get_cbar_axes_tuple
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Histogram_Colorbar_Preset(data: np.ndarray, nbins: int = 256, cmap_norm: Normalize | None = None, cmap_name: str | None = "viridis", position: str = "bottom", figure: Figure | None = None) -> Figure:
    """
    Preset with an Simple axis and a colorbar axis.
    Suitable for visualizing histograms of color data.

    Examples:
        figure = preset.Simple_Colorbar_Preset(data)

    Parameters:
        data: np.ndarray
            Image data.

        nbins:int=256
            Number of bins

        vlim: tuple[float, float] = (0.0, 1.0)
            Min Max values

        cmap_name: str = "viridis"
            colormap name

        position: str = "right"
            color bar position

        figure: Figure | None = None
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

    histogram_axes = figure.gca()
    _cmap_name = cmap_name
    scalar_mappable = ScalarMappable(norm=cmap_norm, cmap=cmap_name)
    histogram_axes.set_xlim((scalar_mappable.norm.vmin, scalar_mappable.norm.vmax))
    histogram_axes.xaxis.set_tick_params(labelbottom=False)
    histogram_axes.set_xlabel(None)
    histogram_axes.tick_params(axis="x", which="both", labelbottom=False)

    divider = make_axes_locatable(histogram_axes)

    colorbar_axes = divider.append_axes(position, size="5%", pad=0.1, sharex=histogram_axes)

    figure.colorbar(scalar_mappable, cax=colorbar_axes, norm=scalar_mappable.norm, cmap=scalar_mappable.cmap, orientation="horizontal")

    bar_xs, bar_width = np.linspace(scalar_mappable.norm.vmin, scalar_mappable.norm.vmax, nbins + 1, retstep=True)
    bar_xcs = bar_xs + bar_width / 2
    bar_colors = scalar_mappable.to_rgba(bar_xcs)
    bar_hs, _ = np.histogram(data, bar_xcs)

    hist_bars = histogram_axes.bar(bar_xcs[:-1], bar_hs, width=bar_width, color=bar_colors)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cmap_norm() -> Normalize:
        return scalar_mappable.norm

    figure.get_cmap_norm = _get_cmap_norm
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_cmap_norm(value: Normalize):
        scalar_mappable.set_norm(value)

    figure.set_cmap_norm = _set_cmap_norm
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cmap_name() -> str:
        return _cmap_name

    figure.get_cmap_name = _get_cmap_name
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_cmap_name(value: str, bad: str | None = "black", over: str | None = None):
        _cmap_name = value
        _cmap = mpl.colormaps[_cmap_name].copy()
        if bad is not None:
            _cmap.set_bad(color=bad)
        if over is not None:
            _cmap.set_over(over)
        scalar_mappable.set_cmap(value)

    figure.set_cmap_name = _set_cmap_name
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_histogram_axes() -> Axes:
        return histogram_axes

    figure.get_histogram_axes = _get_histogram_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes() -> Axes:
        return colorbar_axes

    figure.get_cbar_axes = _get_cbar_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(new_data: np.ndarray):
        bar_xs, bar_width = np.linspace(scalar_mappable.norm.vmin, scalar_mappable.norm.vmax, nbins + 1, retstep=True)
        bar_xcs = bar_xs + bar_width / 2
        bar_colors = scalar_mappable.to_rgba(bar_xcs)
        bar_hs, _ = np.histogram(new_data, bar_xcs)
        for hist_bar, bar_h, bar_x, bar_color in zip(hist_bars, bar_hs, bar_xs, bar_colors):
            hist_bar.set_x(bar_x)
            hist_bar.set_width(bar_width)
            hist_bar.set_height(bar_h)
            hist_bar.set_color(bar_color)

    figure.set_data = _set_data
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset(data: tuple[NDArray[np.float64], NDArray[np.float64]], figure: Figure | None = None) -> Figure:
    """
    Preset with two Imshow axis and Colorbar axis pairs and two axis for plots

    Examples:
        figure = preset.Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset((imageA, imageB))

    Parameters:
        data: tuple[NDArray[np.float64], NDArray[np.float64]]
            Tuple of two images.

        figure: Figure | None = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_images(): list[AxesImage]
            Get imshow image.

        get_images()[index].set_data(image: NDArray[np.float64])
            Set imshow image.

        get_imshow_axes(): list[Axis]
            Get imshow axis.

        get_plot_axes(): list[Axis]
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

    plot_axes_list = [plot_ax_1, plot_ax_2]
    imshow_axes_list = [imshow_ax_a, imshow_ax_b]
    imshow_image_list = [imshow_image_a, imshow_image_b]
    colorbar_axes_list = [colorbar_ax_a, colorbar_ax_b]

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image_list() -> list[AxesImage]:
        return imshow_image_list

    figure.get_image_list = _get_image_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes_list() -> list[Axes]:
        return imshow_axes_list

    figure.get_imshow_axes_list = _get_imshow_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_plot_axes_list() -> list[Axes]:
        return plot_axes_list

    figure.get_plot_axes_list = _get_plot_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes_list() -> list[Axis]:
        return colorbar_axes_list

    figure.get_cbar_axes_list = _get_cbar_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Imshow_Colorbar_Imshow_Colorbar_Plot_Preset(data: tuple[NDArray[np.float64], NDArray[np.float64]], figure: Figure | None = None) -> Figure:
    """
    Preset with two Imshow axis and Colorbar axis pairs and two axis for plots

    Examples:
        figure = preset.Imshow_Colorbar_Imshow_Colorbar_Plot_Preset((imageA, imageB))

    Parameters:
        data: tuple[np.ndarray, np.ndarray]
            Tuple of two images.
        figure: Figure | None = None
            Figure object.

    Returns: Figure
            Figure object with the imshow axis.

    Functions:
        get_images(): list[AxesImage]
            Get imshow image.

        get_images()[index].set_data(image: NDArray[np.float64])
            Set imshow image.

        get_imshow_axes(): list[Axis]
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

    plot_axes = figure.add_subplot(gs1[1])

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

    imshow_axes_list = [imshow_ax_a, imshow_ax_b]
    imshow_images_list = [imshow_image_a, imshow_image_b]
    colorbar_axes_list = [colorbar_ax_a, colorbar_ax_b]

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image_list() -> list[AxesImage]:
        return imshow_images_list

    figure.get_image_list = _get_image_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_imshow_axes_list() -> list[Axes]:
        return imshow_axes_list

    figure.get_imshow_axes_list = _get_imshow_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_plot_axes() -> Axes:
        return plot_axes

    figure.get_plot_axes = _get_plot_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_cbar_axes_list() -> list[Axis]:
        return colorbar_axes_list

    figure.get_cbar_axes_list = _get_cbar_axes_list
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure
