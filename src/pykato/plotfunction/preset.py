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

    zero_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    arg_imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)

    def _get_image() -> AxesImage:
        return arg_imshow_image

    figure.get_image = _get_image

    return figure


def Imshow_Colorbar_Preset(data: np.ndarray, figure: Optional[Figure] = None) -> Figure:

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()
    imshow_image = imshow_ax.imshow(data)
    divider = make_axes_locatable(imshow_ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=colorbar_ax)

    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image

    return figure


def _piFormatter(val, pos) -> str:
    num, den = val.as_integer_ratio()
    if num == 0:
        return ""
    elif den == 1:
        return "$\\pi$" if (num == 1) else "$-\\pi$" if (num == -1) else f"${num}\\pi$"
    else:
        return "$\\frac{{ \\pi }}{{ {} }}$".format(den) if (num == 1) else "$\\frac{{ {}\\pi }}{{ {} }}$".format(den, num)


def Complex_Imshow_2Colorbars_Preset(zdata: np.ndarray, figure: Optional[Figure] = None) -> Figure:

    if figure is None:
        figure = plt.figure()

    imshow_ax = figure.gca()

    arg_image = np.angle(zdata) / np.pi
    mod = np.abs(zdata)
    mod = (mod - np.nanmin(mod)) / (np.nanmax(mod) - np.nanmin(mod))
    mod_image = np.nan_to_num(mod, nan=0.0, posinf=1.0, neginf=0.0)

    zero_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    arg_imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)
    divider = make_axes_locatable(imshow_ax)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(arg_imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_piFormatter))

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.25)
    figure.colorbar(zero_imshow_image, cax=mod_colorbar_ax)

    def _get_image() -> AxesImage:
        return arg_imshow_image

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

    for index, zdata in enumerate(zdatas):

        arg_image = np.angle(zdata) / np.pi
        mod = np.abs(zdata)
        mod = (mod - np.nanmin(mod)) / (np.nanmax(mod) - np.nanmin(mod))
        mod_image = np.nan_to_num(mod, nan=0.0, posinf=1.0, neginf=0.0)

        if index != 0:
            imshow_ax = divider.append_axes("right", size="100%", pad=0.1, sharex=imshow_ax, sharey=imshow_ax)
            imshow_ax.yaxis.set_tick_params(labelleft=False)

        zero_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
        arg_imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)

    arg_colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(arg_imshow_image, cax=arg_colorbar_ax)
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_piFormatter))

    mod_colorbar_ax = divider.append_axes("right", size="5%", pad=0.25)
    figure.colorbar(zero_imshow_image, cax=mod_colorbar_ax)

    return figure
