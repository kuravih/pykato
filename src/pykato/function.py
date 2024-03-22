from typing import Tuple
from enum import Enum, IntEnum, auto
import numpy as np


def generate_coordinates(shape: Tuple[int, int], offset: Tuple[float, float] = (0, 0), cartesian: bool = False, polar: bool = False) -> Tuple[np.ndarray, ...]:
    """Create coordinate grids.

    Examples :
        xx,yy,rr,θθ = generate_coordinates((200,200), True, True)
        xx,yy = generate_coordinates((200,200), cartesian=True)
        rr,θθ = generate_coordinates((200,200), polar=True)

    Parameters :
        shape : Tuple[int, int]
            Image shape.
        offset: Tuple[float, float] = (0, 0)
            offset of origin.
        cartesian: bool = False
            Return cartesian coordinate grid.
        polar: bool = False
            Return polar coordinate grid.

    Returns : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X coordinates.
        Y coordinates.
        R coordinates.
        Theta coordinates.
    """

    _width, _height = shape
    _h_offset, _v_offset = offset
    if not cartesian and not polar:
        xs, ys = np.linspace(0, _width, _width, endpoint=False) + _h_offset, np.linspace(0, _height, _height, endpoint=False) + _v_offset
        return xs, ys
    elif cartesian and not polar:
        xs, ys = np.linspace(0, _width, _width, endpoint=False) + _h_offset, np.linspace(0, _height, _height, endpoint=False) + _v_offset
        xx, yy = np.meshgrid(xs, ys)
        return xx, yy
    elif not cartesian and polar:
        xs, ys = np.linspace(0, _width, _width, endpoint=False) + _h_offset, np.linspace(0, _height, _height, endpoint=False) + _v_offset
        xx, yy = np.meshgrid(xs, ys)
        rr, θθ = np.hypot(xx, yy), np.arctan2(xx, yy)
        return rr, θθ
    else:
        xs, ys = np.linspace(0, _width, _width, endpoint=False) + _h_offset, np.linspace(0, _height, _height, endpoint=False) + _v_offset
        xx, yy = np.meshgrid(xs, ys)
        rr, θθ = np.hypot(xx, yy), np.arctan2(xx, yy)
        return xx, yy, rr, θθ


def checkers(shape: Tuple[int, int], size: Tuple[int, int], offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Create a checker pattern image.

    Example :
        image_checkers = checkers((200,200), (100,100), (50, -50))

    Parameters :
        shape : Tuple[int, int]
            Image shape.
        size : Tuple[int, int]
            Checker size.
        offset: Tuple[int, int] = (0, 0)
            Origin offset.

    Returns : np.ndarray
        Image of the checker pattern.
    """

    checker_width, checker_height = size
    checker_width = 1 if (checker_width == 0) else checker_width
    checker_height = 1 if (checker_height == 0) else checker_height
    xx, yy = generate_coordinates(shape, offset, True)
    xx_steps = (xx // checker_width) % 2
    yy_steps = (yy // checker_height) % 2
    return np.logical_xor(xx_steps, yy_steps)


def sinusoid(shape: Tuple[int, int], period: float, phase: float, angle: float) -> np.ndarray:
    """Create a sinusoidal pattern image.

    Example :
        image_sinusoid = sinusoid((200,200), 10, 20, 60)

    Parameters :
        shape : Tuple[int, int]
            Image shape.
        period : float
            Period of the sinusoid in pixels.
        phase : float
            Phase of the sinusoid in degrees.
        angle : float
            Angle of the sinusoid in degrees.

    Returns : np.ndarray
        Image of the sinusoidal pattern.
    """
    xx, yy = generate_coordinates(shape, cartesian=True)
    pp = (2 * np.pi * (xx * np.cos(np.deg2rad(angle)) + yy * np.sin(np.deg2rad(angle)))) / period
    return (np.sin(pp + np.deg2rad(phase)) + 1) / 2


def vortex(shape: Tuple[int, int], charge: int) -> np.ndarray:
    """Create a Vortex pattern image.

    Example :
        image_vortex = vortex((200,200), 6)

    Parameters :
        shape : Tuple[int, int]
            Image shape.
        charge : int
            Charge of the vortex.

    Returns : np.ndarray
        Image of the vortex pattern.
    """
    image_width, image_height = shape
    xx, yy = generate_coordinates(shape, (0.5 - image_width / 2, 0.5 - image_height / 2), True)
    return (((charge * np.arctan2(xx, yy)) / np.pi) + 1) / 2


def box(shape: Tuple[int, int], size: Tuple[int, int], center: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Create a box pattern image.

    Example :
        image_box = box((200,200), (50,50), (100,100))

    Parameters :
        shape : Tuple[int, int]
            Image shape.
        size : Tuple[int, int]
            Size of the box.
        center: Tuple[int, int] = (0, 0)
            Center of the box.

    Returns : np.ndarray
        Image of the box pattern.
    """
    box_width, box_height = size
    box_x_center, box_y_center = center
    xx, yy = generate_coordinates(shape, (0.5 + box_x_center, 0.5 + box_y_center), True)
    return (-box_width < xx) & (xx < box_width) & (-box_height < yy) & (yy < box_height)


class DOTFProbeDirection(IntEnum):
    RIGHT = 3
    BOTTOM = 6
    LEFT = 9
    TOP = 12

    def to_str(self) -> str:
        return f"{self.value:02d}"


def dotf_probe(shape: Tuple[int, int], size: Tuple[int, int], direction: DOTFProbeDirection) -> np.ndarray:
    """Create a DOTF probe pattern image.

    Example :
        image_dotf_probe = dotf_probe((200,200), (50,50), "12")

    Parameters :
        shape : Tuple[int, int]
            Image shape.
        size : Tuple[int, int]
            Size of the box.
        direction: DOTFProbeDirection
            direction of the probe.

    Returns : np.ndarray
        Image of the dotf probe pattern.
    """
    width, height = shape
    if direction == DOTFProbeDirection.RIGHT:
        center = (-width, -int(height / 2))
    elif direction == DOTFProbeDirection.BOTTOM:
        center = (-int(width / 2), 0)
    elif direction == DOTFProbeDirection.LEFT:
        center = (0, -int(height / 2))
    elif direction == DOTFProbeDirection.TOP:
        center = (-int(width / 2), -height)
    return box(shape, size, center)


class EFCProbeDirection(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()

    def to_str(self) -> str:
        return self.name.lower()


def efc_probe(shape: Tuple[int, int], dξ: float, dη: float, ξc: float, θ: float, direction: EFCProbeDirection) -> np.ndarray:
    """Create a EFC probe pattern image.

    Example :
        TODO:

    Parameters :
        TODO:

    Returns : np.ndarray
        Image of the EFC probe pattern.
    """

    def _efc_probe(_shape: Tuple[int, int], _dξ: float, _dη: float, _ξc: float, _θ: float) -> np.ndarray:
        xx, yy = generate_coordinates(_shape)
        return np.sinc(_dξ * xx) * np.sinc(_dη * yy) * np.sin(_ξc * xx + _θ)

    if direction == EFCProbeDirection.HORIZONTAL:
        return _efc_probe(shape, dξ, dη, ξc, θ)
    else:
        return np.rot90(_efc_probe(shape, dξ, dη, ξc, θ))


def circle(shape: Tuple[int, int], radius: float, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Create a circle pattern image.

    Example :
        image_circle = box((200,200), 50, (100,100))

    Parameters :
        shape : Tuple[int, int]
            Image shape.
        radius : float
            Radius of the circle.
        center : Tuple[float, float] = (0, 0)
            Center of the circle.

    Returns : np.ndarray
        Image of the circle pattern.
    """
    circle_x_center, circle_y_center = center
    xx, yy, rr, __ = generate_coordinates(shape, (0.5 + circle_x_center, 0.5 + circle_y_center), True, True)
    return rr < radius


def Gauss2d_fn(xxyy: Tuple[np.ndarray, np.ndarray], center: Tuple[float, float], offset: float, height: float, width: Tuple[float, float], tilt: float = 0):
    """2d gaussian function (used for fitting).

    Example :
        image_gauss = Gauss2d_fn(TODO: example)

    Parameters :
        xxyy : Tuple[np.ndarray, np.ndarray]
            X Y coordinate arrays.
        center : Tuple[float, float]
            Gaussian center coordinates.
        offset : float
            Offset.
        height : float
            Height of the gaussian peak.
        width : Tuple[float, float]
            2D width of the gaussian.
        tilt : float
            Tilt angle in degrees.

    Returns : np.ndarray
            image of the gaussian.
    """
    xx, yy = xxyy
    x0, y0 = center
    u, v = width
    tilt = np.deg2rad(tilt)
    _a = (np.cos(tilt) ** 2) / (2 * u**2) + (np.sin(tilt) ** 2) / (2 * v**2)
    _b = -(np.sin(2 * tilt)) / (4 * u**2) + (np.sin(2 * tilt)) / (4 * v**2)
    _c = (np.sin(tilt) ** 2) / (2 * u**2) + (np.cos(tilt) ** 2) / (2 * v**2)
    return offset + height * np.exp(-(_a * (xx - x0) ** 2 + 2 * _b * (xx - x0) * (yy - y0) + _c * (yy - y0) ** 2))


def Gauss2d(shape: Tuple[int, int], offset: float = 0, height: float = 1, width: Tuple[float, float] = (3, 3), center: Tuple[float, float] = (0, 0), tilt: float = 0):
    """Create a 2d gauss image.

    Example :
        image_gauss = Gauss2d((200,200), offset=0, height=1, a=(50,25), x=(100,100), tilt=45)

    Parameters :
        shape : Tuple[float, float]
            Image shape.
        offset : float
            DC Offset.
        height : float
            Height of the gaussian peak.
        width : Tuple[float, float]
            2D width of the gaussian.
        center : Tuple[float, float]
            Gaussian center coordinates.
        tilt : float
            Tilt angle in degrees.

    Returns : np.ndarray
        image of the gaussian.
    """
    xx, yy = generate_coordinates(shape, cartesian=True)
    return Gauss2d_fn((xx, yy), center, offset, height, width, tilt)


def polka(shape: Tuple[int, int], radius: float, spacing: Tuple[float, float], offset: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Create a polka dot pattern of 2d gaussian spots.

    Example :
        image_polka = polka((200,200), 3, (40,40), 0)

    Parameters :
        shape : Tuple[float, float]
            Image shape.
        radius : float
            Radius of a dot.
        spacing : Tuple[float, float]
            Spacing between dots
        offset: Tuple[int, int] = (0, 0)
            Origin offset.

    Returns : np.ndarray
        image of the dot pattern.
    """
    width, height = shape
    dot_h_spacing, dot_v_spacing = spacing
    dot_h_offset, dot_v_offset = offset
    xs, ys = np.arange(0, width + dot_h_spacing, dot_h_spacing) - 0.5 + dot_h_offset, np.arange(0, height + dot_v_spacing, dot_v_spacing) - 0.5 + dot_v_offset

    spots = []
    for x in xs:
        for y in ys:
            spots.append(Gauss2d(shape, offset=0, height=1, width=(radius, radius), center=(x, y), tilt=0))

    return np.sum(spots, axis=0)


def Airy_fn(xxyy: Tuple[np.ndarray, np.ndarray], center: Tuple[float, float], radius: float, height: float):
    """2d airy function (used for fitting).

    Example :
        image_airy = Airy_fn(TODO: example)

    Parameters :
        xxyy : Tuple[np.ndarray, np.ndarray]
            X Y coordinate arrays.
        center : Tuple[float, float]
            Gaussian center coordinates.
        radius : float
            Radius.
        A : float
            Height of the airy function.

    Returns : np.ndarray
        Image of the airy function.
    """

    from scipy import special

    xx, yy = xxyy
    x0, y0 = center
    r = radius * np.hypot((xx - x0), (yy - y0))
    return height * np.where(r, (2 * special.jv(1, r) / r) ** 2, 1)


def Airy(shape: Tuple[int, int], center: Tuple[float, float] = (0, 0), radius: float = 1, height: float = 1):
    """Create a 2d Airy disk

    Example :
        image_airy = Airy2d(TODO: example)

    Parameters :
        shape : Tuple[float, float]
            Image shape.
        center : Tuple[float, float]
            Gaussian center coordinates.
        radius : float
            Radius.
        height : float
            Height of the airy function.

    Returns : np.ndarray
        Image of the Airy disk.
    """
    xx, yy = generate_coordinates(shape, cartesian=True)
    return Airy_fn((xx, yy), center, radius, height)


def PSF_to_OTF(psf: np.ndarray, axes=None) -> np.ndarray:
    """Calculate the Optical Transfer Function (OTF) from a PSF

    Example :
        otf = PSF_to_OTF(psf_image)

    Parameters :
        psf : np.ndarray
            PSF image.

    Returns : np.ndarray
        Complex OTF image.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf, axes=axes), axes=axes), axes=axes)


def hsort(_xy, ascend: bool = True):
    if ascend:
        return np.argsort((_xy[:, 0]))
    else:
        return np.flip(np.argsort((_xy[:, 0])))


def vsort(_xy, ascend: bool = True):
    if ascend:
        return np.argsort((_xy[:, 1]))
    else:
        return np.flip(np.argsort((_xy[:, 1])))


def xy_sort(_xy, shape: Tuple[int, int], hascend: bool = True, vascend: bool = True):
    indices = []
    hsorted_indices = np.reshape(hsort(_xy, hascend), shape)
    for _indices in hsorted_indices:
        vsorted_indices = vsort(_xy[_indices], vascend)
        indices.append(_indices[vsorted_indices])
    return np.array(indices).flatten()


def Linear2d_fn(x, y, a, b, c):
    return a * x + b * y + c


def linear_2d_lsq_fit(data):
    from scipy.optimize import curve_fit

    def _linear_2d(M, a, b, c):
        x, y = M
        return Linear2d_fn(x, y, a, b, c)

    mask = np.isnan(data)
    x_coordinates, y_coordinates = generate_coordinates(data.shape, cartesian=True)
    z_data = data.copy()
    x_coordinates = np.ma.masked_array(x_coordinates, mask=mask)
    y_coordinates = np.ma.masked_array(y_coordinates, mask=mask)
    z_data = np.ma.masked_array(z_data, mask=mask)

    xy_coordinates = np.vstack((x_coordinates.compressed().ravel(), y_coordinates.compressed().ravel()))
    zdata = z_data.compressed()

    guess_prms = [0, 0, 0]
    popt, pcov = curve_fit(_linear_2d, xy_coordinates, zdata, guess_prms)
    tip, tilt, piston = popt

    return tip, tilt, piston


def dphtf_to_command(_deltatf, _locations):
    """Convert delta phase transfer function to a dm command

    Parameters
    ----------
        _deltatf : np.ndarray
            Differential transfer function image
        _locations :  np.ndarray (2, nact)
            Actuator locations

    Returns
    -------
        cmd : nd.ndarray (nact)
            Actuator command

    """
    from scipy.interpolate import RegularGridInterpolator

    grid_x, grid_y = np.arange(_deltatf.shape[0]), np.arange(_deltatf.shape[1])
    surface_fn = RegularGridInterpolator((grid_x, grid_y), _deltatf)
    cmd = np.zeros(_locations.shape[0])
    for index, px in enumerate(_locations):
        cmd[index] = surface_fn((px[0], px[1]))
    return cmd - np.nanmean(cmd)


def calculate_command(shape, dotf_images, px_location_arrays):
    command = np.full(shape, np.nan)
    slm_circle = circle(shape, shape[0] / 2, (-shape[0] / 2, -shape[1] / 2))
    commands = []
    for index, (dotf_image, px_location_array) in enumerate(zip(dotf_images, px_location_arrays)):
        command[slm_circle] = dphtf_to_command(np.angle(dotf_image), px_location_array)
        if index == 0:
            command[:, :22] = np.nan
        elif index == 1:
            command[:22, :] = np.nan
        elif index == 2:
            command[:, -22:] = np.nan
        else:
            command[-22:, :] = np.nan
        commands.append(command.copy())

    mean_command = np.nanmean(commands, axis=0)
    tip, tilt, piston = linear_2d_lsq_fit(mean_command)
    x_coordinates, y_coordinates = generate_coordinates(mean_command.shape, cartesian=True)

    return mean_command - x_coordinates * tip - y_coordinates * tilt - piston


# # =====================================================================================================================
# def Fit2d(_z, _xy, _func, _guess):
#   """ Fit an image to a 2d function

#   Parameters
#   ----------
#     _z : array_like
#       surface
#     _xy : (array_like, array_like)
#       x,y images of the same shape as z

#   Returns
#   -------
#     popt : array
#       dark subtracted averaged image
#     pcov : 2-D array
#       The estimated approximate covariance of popt.
#   """

#   from scipy.optimize import curve_fit

#   xx, yy = _xy

#   assert (xx.shape == yy.shape == _z.shape), 'x or y not same shape as z'

#   def _fitFunc(M, *args):
#     _x, _y = M
#     return _func(_x, _y, *args)

#   xdata = np.vstack((xx.ravel(), yy.ravel()))
#   ydata = _z.ravel()
#   popt, pcov = curve_fit(_fitFunc, xdata, ydata, _guess)

#   return popt, pcov
# # =====================================================================================================================
