from typing import Tuple, Optional
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def Imshow_Preset(data: np.ndarray, figure: Optional[Figure] = None) -> Figure:
    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    imshow_image = imshow_ax.imshow(data)
    imshow_ax.invert_yaxis()

    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image

    return figure


def Complex_Imshow_Preset(zdata: np.ndarray, figure: Optional[Figure] = None) -> Figure:
    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()

    arg_image = np.angle(zdata) / np.pi
    mod = np.abs(zdata)
    mod = (mod - np.nanmin(mod)) / (np.nanmax(mod) - np.nanmin(mod))
    mod_image = np.nan_to_num(mod, nan=0.0, posinf=1.0, neginf=0.0)

    _bg_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)

    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image

    return figure


def Imshow_Colorbar_Preset(data: np.ndarray, figure: Optional[Figure] = None) -> Figure:
    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    imshow_image = imshow_ax.imshow(data)
    imshow_ax.invert_yaxis()
    divider = make_axes_locatable(imshow_ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=colorbar_ax)

    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image

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


def _complex_to_plot_arg_mod(zdata):
    arg_image = np.angle(zdata) / np.pi
    mod = np.abs(zdata)
    mod = (mod - np.nanmin(mod)) / (np.nanmax(mod) - np.nanmin(mod))
    mod_image = np.nan_to_num(mod, nan=0.0, posinf=1.0, neginf=0.0)
    return arg_image, mod_image


def Complex_Imshow_2Colorbars_Preset(zdata: np.ndarray, figure: Optional[Figure] = None) -> Figure:
    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()

    arg_image, mod_image = _complex_to_plot_arg_mod(zdata)

    _bg_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)
    divider = make_axes_locatable(imshow_ax)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_piFormatter))
    arg_colorbar_ax.set_title("arg", size=10)

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.4)
    figure.colorbar(_bg_imshow_image, cax=mod_colorbar_ax)
    mod_colorbar_ax.set_title("mod", size=10)

    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image

    return figure


def ImageGrid_Preset(datas: Tuple[np.ndarray], figure: Optional[Figure] = None) -> Figure:

    if figure is None:
        figure = plt.figure()

    imshow_axes = ImageGrid(figure, 111, nrows_ncols=(1, len(datas)), axes_pad=0.1)

    for imshow_axis, data in zip(imshow_axes, datas):
        imshow_axis.imshow(data)

    return figure


def ImageGrid_Colorbar_Preset(datas: Tuple[np.ndarray], figure: Optional[Figure] = None) -> Figure:

    if figure is None:
        figure = plt.figure()

    imshow_axes = ImageGrid(figure, 111, nrows_ncols=(1, len(datas)), axes_pad=0.1, share_all=True, label_mode="L", cbar_location="right", cbar_mode="single")

    for imshow_axis, data in zip(imshow_axes, datas):
        imshow_image = imshow_axis.imshow(data)

    imshow_axes.cbar_axes[0].colorbar(imshow_image)

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

    return figure
