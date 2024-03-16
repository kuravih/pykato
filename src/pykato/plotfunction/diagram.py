from typing import Tuple, Optional
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure

from .gridspec_layout import Image_Layout, Image_Colorbar_Layout, Image_Colorbar_Colorbar_Layout


def Image_Diagram(data, figure: Optional[Figure] = None, *args, **kwargs) -> Figure:
    """
    Create an image diagram.
    Add following methods to the diagram figure object.
    Requires one axis object in the figure.

    Methods
    -------
        imshow_set_clim(clim:Tuple[float, float])
            Set color limits

        imshow_set_cmap(cmap:str)
            Set color map

        imshow_set_bad(color:str)
            Set diagram colormap bad color

        imshow_set_data(image:np.ndarray)
            Set diagram data
    """
    if figure is None:
        figure = Image_Layout(*args, **kwargs)

    try:
        (imshow_ax,) = figure.get_axes()
    except Exception as err:
        print(f"Could not retrieve axis from given figure : {figure}")
        raise err

    imshow_image = imshow_ax.imshow(data)
    cmap = imshow_image.get_cmap()
    imshow_ax.invert_yaxis()

    # -----------------------------------------------------------------------------------------------------------------
    figure.imshow_set_clim = imshow_image.set_clim
    # -----------------------------------------------------------------------------------------------------------------
    figure.imshow_set_cmap = imshow_image.set_cmap

    # -----------------------------------------------------------------------------------------------------------------
    def _set_bad(_color) -> None:
        cmap.set_bad(color=_color)
        imshow_image.set_cmap(cmap)

    figure.imshow_set_bad = _set_bad
    # -----------------------------------------------------------------------------------------------------------------
    figure.imshow_set_data = imshow_image.set_data
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Image_Colorbar_Diagram(data, figure: Optional[Figure] = None, *args, **kwargs) -> Figure:
    """
    Create an image + colorbar diagram.
    Add following methods to the diagram figure object.
    Requires two axis objects in the figure.

    Methods
    -------
        imshow_set_clim(clim:Tuple[float, float])
            Set color limits

        imshow_set_cmap(cmap:str)
            Set color map

        imshow_set_bad(color:str)
            Set diagram colormap bad color

        imshow_set_data(image:np.ndarray)
            Set diagram data
    """
    if figure is None:
        figure = Image_Colorbar_Layout(*args, **kwargs)

    try:
        imshow_ax, colorbar_ax = figure.get_axes()
    except Exception as err:
        print(f"Could not retrieve axes from given figure : {figure}")
        raise err

    imshow_image = imshow_ax.imshow(data)
    cmap = imshow_image.get_cmap()
    imshow_ax.invert_yaxis()
    figure.colorbar(imshow_image, cax=colorbar_ax, orientation="vertical")

    # -----------------------------------------------------------------------------------------------------------------
    figure.imshow_set_clim = imshow_image.set_clim
    # -----------------------------------------------------------------------------------------------------------------
    figure.imshow_set_cmap = imshow_image.set_cmap

    # -----------------------------------------------------------------------------------------------------------------
    def _set_bad(_color) -> None:
        cmap.set_bad(color=_color)
        imshow_image.set_cmap(cmap)

    figure.imshow_set_bad = _set_bad
    # -----------------------------------------------------------------------------------------------------------------
    figure.imshow_set_data = imshow_image.set_data
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def _piFormatter(val, pos) -> str:
    num, den = val.as_integer_ratio()
    if num == 0:
        return ""
    elif den == 1:
        return "$\\pi$" if (num == 1) else "${}\\pi$".format(num)
    else:
        return "$\\frac{{ \\pi }}{{ {} }}$".format(den) if (num == 1) else "$\\frac{{ {}\\pi }}{{ {} }}$".format(den, num)


def Complex_Diagram(zdata, figure=None, *args, **kwargs):
    """
    Create an image + two colorbar diagram.
    Add following methods to the diagram figure object.
    Requires three axis objects in the figure.

    Methods
    -------
        mod_imshow_set_cmap(cmap:str)
            Set diagram modulus color map

        arg_imshow_set_cmap(cmap:str)
            Set diagram argument color map

        mod_imshow_set_bad(color:str)
            Set diagram modulus colormap bad color

        arg_imshow_set_bad(color:str)
            Set diagram argument colormap bad color

        set_data(image:np.ndarray)
            Set diagram data
    """
    if figure is None:
        figure = Image_Colorbar_Colorbar_Layout(*args, **kwargs)

    try:
        imshow_ax, arg_colorbar_ax, mod_colorbar_ax = figure.get_axes()
    except Exception as err:
        print(f"Could not retrieve axes from given figure : {figure}")
        raise err

    arg_image = np.angle(zdata) / np.pi
    mod = np.abs(zdata)
    mod = (mod - np.nanmin(mod)) / (np.nanmax(mod) - np.nanmin(mod))
    mod_image = np.nan_to_num(mod, nan=0.0, posinf=1.0, neginf=0.0)
    mod_imshow_image = imshow_ax.imshow(np.full(zdata.shape, 0, dtype=float), "gray", vmin=0, vmax=1)
    arg_imshow_image = imshow_ax.imshow(arg_image, alpha=mod_image, cmap="hsv", vmin=-1, vmax=1)
    imshow_ax.invert_yaxis()

    figure.colorbar(arg_imshow_image, cax=arg_colorbar_ax, orientation="vertical")
    arg_colorbar_ax.yaxis.set_major_formatter(FuncFormatter(_piFormatter))
    arg_cmap = arg_imshow_image.get_cmap()

    figure.colorbar(mod_imshow_image, cax=mod_colorbar_ax, orientation="vertical")
    mod_cmap = mod_imshow_image.get_cmap()

    # -----------------------------------------------------------------------------------------------------------------
    figure.mod_imshow_set_cmap = mod_imshow_image.set_cmap
    # -----------------------------------------------------------------------------------------------------------------
    figure.arg_imshow_set_cmap = arg_imshow_image.set_cmap

    # -----------------------------------------------------------------------------------------------------------------
    def _mod_set_bad(_color) -> None:
        mod_cmap.set_bad(color=_color)
        mod_imshow_image.set_cmap(mod_cmap)

    figure.mod_imshow_set_bad = _mod_set_bad

    # -----------------------------------------------------------------------------------------------------------------
    def _arg_set_bad(_color) -> None:
        arg_cmap.set_bad(color=_color)
        arg_imshow_image.set_cmap(arg_cmap)

    figure.arg_imshow_set_bad = _arg_set_bad

    # -----------------------------------------------------------------------------------------------------------------
    def _set_data(zdata: np.ndarray) -> None:
        arg_image = np.angle(zdata) / np.pi
        mod = np.abs(zdata)
        mod = (mod - np.nanmin(mod)) / (np.nanmax(mod) - np.nanmin(mod))
        mod_image = np.nan_to_num(mod, nan=0.0, posinf=1.0, neginf=0.0)
        arg_imshow_image.set_data(arg_image)
        arg_imshow_image.set_alpha(mod_image)

    figure.set_data = _set_data
    # -----------------------------------------------------------------------------------------------------------------

    return figure
