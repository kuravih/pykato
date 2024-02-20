from typing import Tuple, Optional
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.colors import hsv_to_rgb

from .fixed_layout import Image_Layout, Image_Colorbar_Layout, Image_Colorbar_Colorbar_Layout

from matplotlib.figure import Figure


def Image_Diagram(data, figure: Optional[Figure] = None, colLim: Tuple[float, float] = (0, 10), *args, **kwargs) -> Figure:
    """
    Add following methods to figure object.
    Requires one axis object in the figure.

    Methods
    -------
        set_data(image:np.ndarray, colLim: Optional[Tuple[float, float]] = None) -> None:
            Set diagram data

        set_cmap(_cmap) -> None:
            Set diagram colormap

        set_bad(_color) -> None:
            Set diagram colormap bad color

        set_title(title:str) -> None:
            Set diagram title
    """
    if figure is None:
        figure = Image_Layout(*args, **kwargs)

    try:
        (imshow_ax,) = figure.get_axes()
    except Exception as err:
        print(f"Could not retrieve axes from given figure : {figure}")
        raise err

    imshow_ax.imshow(data)
    imshow_image = imshow_ax.get_images()[0]
    cmap = imshow_image.get_cmap()
    imshow_ax.invert_yaxis()

    def _set_data(_image: np.ndarray, colLim: Optional[Tuple[float, float]] = None) -> None:
        imshow_image.set_data(_image)
        if colLim is not None:
            imshow_image.set_clim(colLim[0], colLim[1])

    figure.set_data = _set_data

    def _set_cmap(_cmap) -> None:
        imshow_image.set_cmap(_cmap)

    figure.set_cmap = _set_cmap

    def _set_title(_title: str) -> None:
        imshow_ax.set_title(_title)

    figure.set_title = _set_title

    def _set_bad(_color) -> None:
        cmap.set_bad(color=_color)
        imshow_image.set_cmap(cmap)

    figure.set_bad = _set_bad

    return figure


def Image_Colorbar_Diagram(data, figure: Optional[Figure] = None, colLim: Tuple[float, float] = (0, 10), colN: int = 10, *args, **kwargs) -> Figure:
    """
    Add following methods to figure object.
    Requires two axis object in the figure.

    Methods
    -------
        set_data(image:np.ndarray, colLim: Optional[Tuple[float, float]] = None) -> None:
            Set diagram data

        set_cmap(cmap) -> None:
            Set diagram colormap

        set_bad(color) -> None:
            Set diagram colormap bad color

        set_title(title:str) -> None:
            Set diagram title
    """

    if figure is None:
        figure = Image_Colorbar_Layout(*args, **kwargs)

    try:
        imshow_ax, colorbar_ax = figure.get_axes()
    except Exception as err:
        print(f"Could not retrieve axes from given figure : {figure}")
        raise err

    imshow_ax.imshow(data)
    imshow_image = imshow_ax.get_images()[0]
    cmap = imshow_image.get_cmap()
    cArray, dCol = np.linspace(colLim[0], colLim[1], num=colN, retstep=True, endpoint=True)
    colGradient = np.vstack((cArray, cArray))
    colorbar_ax.imshow(np.rot90(colGradient, 1), aspect="auto", extent=(0, 1, colLim[0], colLim[1]))
    colorbar_image = colorbar_ax.get_images()[0]
    imshow_ax.invert_yaxis()
    colorbar_ax.xaxis.set_visible(False)
    colorbar_ax.yaxis.tick_right()

    def _set_data(_image, colLim: Optional[Tuple[float, float]] = None) -> None:
        imshow_image.set_data(_image)
        if colLim is not None:
            colorbar_image.set_extent((0, 1, colLim[0], colLim[1]))
            imshow_image.set_clim(colLim[0], colLim[1])

    figure.set_data = _set_data

    def _set_cmap(_cmap) -> None:
        imshow_image.set_cmap(_cmap)
        colorbar_image.set_cmap(_cmap)

    figure.set_cmap = _set_cmap

    def _set_title(_title: str) -> None:
        imshow_ax.set_title(_title)

    figure.set_title = _set_title

    def _set_bad(_color) -> None:
        cmap.set_bad(color=_color)
        imshow_image.set_cmap(cmap)

    figure.set_bad = _set_bad

    return figure


def _complex_to_rgb(z, modMax: Optional[float] = None, argMax: float = 2 * np.pi, theme: str = "dark"):
    hsv = np.zeros(z.shape + (3,), dtype="float")
    mod, arg = np.abs(z), np.angle(z)
    if modMax is None:
        modMax = np.max(mod)
    hsv[..., 0] = (arg / (argMax)) % 1
    if theme == "dark":
        hsv[..., 1] = 1
        hsv[..., 2] = np.clip(mod / modMax, 0, 1)
    else:
        hsv[..., 1] = np.clip(mod / modMax, 0, 1)
        hsv[..., 2] = 1
    return hsv_to_rgb(hsv)


def _piFormatter(val, pos) -> str:
    num, den = val.as_integer_ratio()
    if num == 0:
        return ""
    elif den == 1:
        return "$\\pi$" if (num == 1) else "${}\\pi$".format(num)
    else:
        return "$\\frac{{ \\pi }}{{ {} }}$".format(den) if (num == 1) else "$\\frac{{ {}\\pi }}{{ {} }}$".format(den, num)


def Complex_Diagram(zdata, figure: Optional[Figure] = None, modLim: Tuple[float, float] = (0, 10), modN: int = 100, argLim: Tuple[float, float] = (0, 2), argN: int = 100, *args, **kwargs) -> Figure:
    """
    Add following methods to figure object.
    Requires three axis object in the figure.

    Methods
    -------
        set_data(image:np.ndarray, colLim: Optional[Tuple[float, float]] = None) -> None:
            Set diagram data

        set_cmap(_cmap) -> None:
            Set diagram colormap

        set_bad(_color) -> None:
            Set diagram colormap bad color

        set_title(title:str) -> None:
            Set diagram title
    """

    if figure is None:
        figure = Image_Colorbar_Colorbar_Layout(*args, **kwargs)

    try:
        imshow_ax, colorbar_mod_ax, colorbar_arg_ax = figure.get_axes()
    except Exception as err:
        print(f"Could not retrieve axes from given figure : {figure}")
        raise err

    imshow_ax.imshow(_complex_to_rgb(zdata), cmap="hsv")
    imshow_image = imshow_ax.get_images()[0]
    cmap = imshow_image.get_cmap()

    argArray, dArg = np.linspace(argLim[0], argLim[1], num=argN, retstep=True, endpoint=True)
    argGradient = np.vstack((argArray, argArray))
    colorbar_arg_ax.imshow(np.rot90(argGradient, 1), aspect="auto", cmap="hsv", extent=(0, 1, argLim[0], argLim[1]))
    colorbar_arg_image = colorbar_arg_ax.get_images()[0]

    modArray, dMod = np.linspace(modLim[0], modLim[1], num=modN, retstep=True, endpoint=True)
    modGradient = np.vstack((modArray, modArray))
    colorbar_mod_ax.imshow(np.rot90(modGradient, 3), aspect="auto", cmap="binary", extent=(0, 1, modLim[0], modLim[1]))
    colorbar_mod_image = colorbar_mod_ax.get_images()[0]

    imshow_ax.invert_yaxis()
    colorbar_mod_ax.xaxis.set_visible(False)
    colorbar_mod_ax.yaxis.tick_right()
    colorbar_arg_ax.set_title("$\phi$", y=-0.1)

    colorbar_arg_ax.xaxis.set_visible(False)
    colorbar_arg_ax.yaxis.tick_right()
    colorbar_arg_ax.yaxis.set_major_locator(MultipleLocator(base=0.5))
    colorbar_arg_ax.yaxis.set_major_formatter(FuncFormatter(_piFormatter))
    colorbar_mod_ax.set_title("r", y=-0.1)

    def _set_data(_zdata, modMax: Optional[float] = None, *args, **kwargs) -> None:
        imshow_image.set_data(_complex_to_rgb(_zdata, modMax, *args, **kwargs))
        colorbar_mod_image.set_extent((0, 1, 0, modMax))

    figure.set_data = _set_data

    def _set_title(_title: str) -> None:
        imshow_ax.set_title(_title)

    figure.set_title = _set_title

    def _set_bad(_color) -> None:
        cmap.set_bad(color=_color)
        imshow_image.set_cmap(cmap)

    figure.set_bad = _set_bad

    return figure
