import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy import special, ndimage
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.restoration import unwrap_phase
from PIL import Image, ImageDraw, ImageFont
from importlib.resources import files
import datetime


def generate_coordinates(shape: tuple[int, int], offset: tuple[float, float] | None = None, cartesian: bool = False, polar: bool = False) -> tuple[np.ndarray, ...]:
    """
    Create coordinate grids.

    Examples:
        xx, yy, rr, θθ = generate_coordinates((200, 200), cartesian=True, polar=True)
        xx, yy = generate_coordinates((200, 200), cartesian=True)
        rr, θθ = generate_coordinates((200, 200), polar=True)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        offset: tuple[float, float] | None = None
            Offset of origin.
            if None origin will be placed at the gird center.
            Use (0,0) to place origin at the corner
        cartesian: bool = False
            Return cartesian coordinate grid.
        polar: bool = False
            Return polar coordinate grid.

    Returns: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X coordinates.
        Y coordinates.
        R coordinates.
        Theta coordinates.
    """

    width, height = shape
    if offset is None:
        offset = (-(width - 1) / 2, -(height - 1) / 2)
    h_offset, v_offset = offset
    xs, ys = np.arange(0, width) + h_offset, np.arange(0, height) + v_offset
    if not cartesian and not polar:
        return xs, ys
    elif cartesian and not polar:
        xx, yy = np.meshgrid(xs, ys)
        return xx, yy
    elif not cartesian and polar:
        xx, yy = np.meshgrid(xs, ys)
        rr, θθ = np.hypot(xx, yy), np.arctan2(yy, xx)
        return rr, θθ
    else:
        xx, yy = np.meshgrid(xs, ys)
        rr, θθ = np.hypot(xx, yy), np.arctan2(yy, xx)
        return xx, yy, rr, θθ


def gradient(shape: tuple[int, int], angle: float = 0, origin: tuple[float, float] | None = None) -> np.ndarray:
    """
    Create a gradient pattern image bounded within [-1, 1].

    Example:
        image_gradient = gradient((200,200), 45)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        angle: float
            Angle of the gradient in radians
        origin: tuple[float, float] | None = None
            origin of gradient.
            if None origin will be placed at the gird center.
            Use (0,0) to place origin the corner

    Returns: np.ndarray
        Image of the gradient pattern.
    """
    xx, yy = generate_coordinates(shape, origin, cartesian=True)
    grad = xx * np.cos(angle) + yy * np.sin(angle)
    return grad


def checkers(shape: tuple[int, int], size: tuple[int, int], offset: tuple[int, int] | None = None, color1: float = 1.0, color2: float = 0.0) -> np.ndarray:
    """
    Create a checker pattern image.

    Example:
        image_checkers = checkers((200,200), (100,100), (50, -50))

    Parameters:
        shape: tuple[int, int]
            Image shape.
        size: tuple[int, int]
            Checker size.
        offset: tuple[float, float] | None = None
            Offset of origin.
            if None origin will be placed at the gird center.
            Use (0,0) to place origin the corner

    Returns: np.ndarray
        Image of the checker pattern.
    """

    checker_width, checker_height = size
    xx, yy = generate_coordinates(shape, offset, True)
    xx_steps = (xx // checker_width) % 2
    yy_steps = (yy // checker_height) % 2
    return np.logical_xor(xx_steps, yy_steps)


def sinusoid(shape: tuple[int, int], period: float, angle: float = 0, phase: float = 0) -> np.ndarray:
    """
    Create a sinusoidal pattern image.

    Example:
        image_sinusoid = sinusoid((200,200), 10, 20, 60)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        period: float
            Period of the sinusoid in pixels.
        angle: float
            Angle of the sinusoid in radians.
        phase: float
            Phase of the sinusoid in radians.

    Returns: np.ndarray
        Image of the sinusoidal pattern.
        The mean is 0.0 and the amplitude is 1.0 (i.e min = -1.0, max = +1.0)
    """
    pp = 2 * np.pi * gradient(shape, angle)
    ff = 1 / period
    return np.sin(ff * pp + phase)


def vortex(shape: tuple[int, int], charge: int = 1, angle: float = 0.0, center: tuple[float, float] | None = None) -> np.ndarray:
    """
    Create a Vortex pattern image.

    Example:
        image_vortex = vortex((200,200), 6)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        charge: int
            Charge of the vortex.
        angle: float
            Angle of the vortex.
        center: tuple[float, float] | None = None
            Center of the vortex.
            if None origin will be placed at the gird center.
            Use (0,0) to place origin the corner

    Returns: np.ndarray
        Image of the vortex pattern.
    """
    xx, yy = generate_coordinates(shape, center, cartesian=True)
    # Apply rotation
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    xx_rot = cos_a * xx - sin_a * yy
    yy_rot = sin_a * xx + cos_a * yy
    return charge * np.arctan2(yy_rot, xx_rot)


def box(shape: tuple[int, int], size: tuple[int, int], center: tuple[float, float] | None = None) -> np.ndarray:
    """
    Create a box pattern image.

    Example:
        image_box = box((200,200), (50,50), (100,100))

    Parameters:
        shape: tuple[int, int]
            Image shape.
        size: tuple[int, int]
            Size of the box.
        center: tuple[float, float] | None = None
            Center of the box.
            if None center will be placed at the gird center.
            Use (0,0) to place center at the corner

    Returns: np.ndarray
        Image of the box pattern.
    """
    box_width, box_height = size
    xx, yy = generate_coordinates(shape, center, cartesian=True)
    return (-box_width < xx) & (xx < box_width) & (-box_height < yy) & (yy < box_height)


def disk(shape: tuple[int, int], radius: float, center: tuple[float, float] | None = None) -> np.ndarray:
    """
    Create a disk pattern image.

    Example:
        image_disk = disk((200,200), 50, (100,100))

    Parameters:
        shape: tuple[int, int]
            Image shape.
        radius: float
            Radius of the disk.
        center: tuple[float, float] | None = None
            Center of the disk.
            if None center will be placed at the gird center.
            Use (0,0) to place center at the corner

    Returns: np.ndarray
        Image of the disk pattern.
    """
    rr, _ = generate_coordinates(shape, center, polar=True)
    return rr <= radius


def chord(shape: tuple[int, int], radius: float, percent: float, angle: float = 0, center: tuple[float, float] | None = None) -> np.ndarray:
    """
    Create a chord (disk portion) pattern image.

    Example:
        image_chord = chord((200,200), 50, -0.2, 3 * np.pi / 2, (100,100))

    Parameters:
        shape: tuple[int, int]
            Image shape.
        radius: float
            Radius of the disk.
        percent: float
            Percentage of radius to cutoff.
        angle: float
            angle to cutoff.
        center: tuple[float, float] | None = None
            Center of the disk.
            if None center will be placed at the gird center.
            Use (0,0) to place center at the corner

    Returns: np.ndarray
        Image of the chord pattern.
    """
    disk_image = disk(shape, radius, center)
    grad_image = gradient(shape, angle, center)
    temp_image = disk_image * grad_image
    temp_min, temp_ptp = np.min(temp_image), np.ptp(temp_image)
    cutoff = temp_min + temp_ptp * percent
    return (grad_image >= cutoff) * disk_image


def gauss2d_fn(xx_yy: tuple[np.ndarray, np.ndarray], center: tuple[float, float], offset: float, height: float, width: tuple[float, float], tilt: float = 0):
    """
    2d gaussian function (used for fitting).

    Example:
        image_gauss = Gauss2d_fn(TODO: example)

    Parameters:
        xx_yy: tuple[np.ndarray, np.ndarray]
            X Y coordinate arrays.
        center: tuple[float, float] | None = None
            Center of the disk.
            if None center will be placed at the gird center.
            Use (0,0) to place center at the corner
        offset: float
            Offset.
        height: float
            Height of the gaussian peak.
        width: tuple[float, float]
            2D width of the gaussian.
        tilt: float
            Tilt angle in radians.

    Returns: np.ndarray
            image of the gaussian.
    """
    xx, yy = xx_yy
    x0, y0 = center
    u, v = width
    _a = (np.cos(tilt) ** 2) / (2 * u**2) + (np.sin(tilt) ** 2) / (2 * v**2)
    _b = -(np.sin(2 * tilt)) / (4 * u**2) + (np.sin(2 * tilt)) / (4 * v**2)
    _c = (np.sin(tilt) ** 2) / (2 * u**2) + (np.cos(tilt) ** 2) / (2 * v**2)
    return offset + height * np.exp(-(_a * (xx - x0) ** 2 + 2 * _b * (xx - x0) * (yy - y0) + _c * (yy - y0) ** 2))


def gauss2d(shape: tuple[int, int], offset: float = 0, height: float = 1, width: tuple[float, float] = (3, 3), center: tuple[float, float] = (0, 0), tilt: float = 0):
    """
    Create a 2d gauss image.

    Example:
        image_gauss = Gauss2d((200,200), offset=0, height=1, a=(50,25), x=(100,100), tilt=45)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        offset: float
            DC Offset.
        height: float
            Height of the gaussian peak.
        width: tuple[float, float]
            2D width of the gaussian.
        center: tuple[float, float]
            Gaussian center coordinates.
        tilt: float
            Tilt angle in degrees.

    Returns: np.ndarray
        image of the gaussian.
    """
    xx, yy = generate_coordinates(shape, (0, 0), cartesian=True)
    return gauss2d_fn((xx, yy), center, offset, height, width, tilt)


def polka(shape: tuple[int, int], radius: float, spacing: tuple[float, float], offset: tuple[float, float] = (0, 0)) -> np.ndarray:
    """
    Create a polka dot pattern of 2d gaussian spots.

    Example:
        image_polka = polka((200,200), 3, (40,40), 0)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        radius: float
            Radius of a dot.
        spacing: tuple[float, float]
            Spacing between dots
        offset: tuple[float, float] | None = None
            Offset of origin.
            if None origin will be placed at the gird center.
            Use (0,0) to place origin the corner

    Returns: np.ndarray
        image of the dot pattern.
    """
    width, height = shape
    dot_h_spacing, dot_v_spacing = spacing
    h_offset, v_offset = offset[0] % dot_h_spacing, offset[1] % dot_v_spacing
    xs, ys = np.arange(-dot_h_spacing, width + dot_h_spacing, dot_h_spacing) + h_offset, np.arange(-dot_v_spacing, height + dot_v_spacing, dot_v_spacing) + v_offset

    spots = []
    for x in xs:
        for y in ys:
            spots.append(disk(shape, radius, center=(-x, -y)))

    return np.sum(spots, axis=0)


def register(shape: tuple[int, int], count: tuple[int, int], radius: float, spacing: tuple[float, float], center: tuple[float, float] = (0, 0), normalize=True) -> np.ndarray:
    """
    Create a polka dot pattern of 2d gaussian spots.

    Example:
        image_register = register((200,200), (4,4), 3, (40,40), 0)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        count: tuple[int, int]
            Number of dots.
        radius: float
            Radius of a dot.
        spacing: tuple[float, float]
            Spacing between dots
        center: tuple[int, int] = (0, 0)
            Origin center.

    Returns: np.ndarray
        image of the registration dot pattern.
    """
    dot_h_spacing, dot_v_spacing = spacing
    dot_h_center, dot_v_center = center
    xs, ys = dot_h_spacing * (np.arange(0, count[0]) + 0.5) + dot_h_center - dot_h_spacing * count[0] / 2, dot_v_spacing * (np.arange(0, count[1]) + 0.5) + dot_v_center - dot_v_spacing * count[1] / 2

    spots = []
    for x in xs:
        for y in ys:
            if radius == 0:
                points = np.zeros((shape))
                points[int(np.floor(x)), int(np.floor(y))] = 1
                spots.append(points)
            else:
                spots.append(gauss2d(shape, offset=0, height=1, width=(radius, radius), center=(x, y), tilt=0))

    image = np.sum(spots, axis=0)
    if normalize:
        return image / np.max(image)
    else:
        return image


def text(shape: tuple[int, int], string: str, position: tuple[float, float] | None = None, font_size: int | None = None) -> np.ndarray:
    """
    Create an image of a string of characters.

    Example:
        image_character = character((200,200), '4', (100,100), (100,100))

    Parameters:
        shape: tuple[int, int]
            Image shape.
        string: str
            string to be rendered.
        position: tuple[float, float] | None
            Position of the character. If None, centers the character.
        font_size: int | None
            Size of the string. If None, auto-sizes to fit the image.

    Returns: np.ndarray
        Image of the character string.
    """

    img = Image.new("L", shape, color=0)  # Create a blank image for mask
    draw = ImageDraw.Draw(img)

    if font_size is None:
        font_size = min(shape)

    font = ImageFont.truetype(files("pykato.resource") / "ProggyClean.ttf", font_size)

    if position is None:
        position = (shape[0] / 2, shape[1] / 2)

    draw.text(position, string, align="center", font=font, fill=255, anchor="mm")  # Draw string

    return np.array(img) / 255.0


def preroll(shape: tuple[int, int], count: int, percent: float = 0, font_size: int | None = None) -> np.ndarray:
    """
    Create a pre roll image of a count down.

    Example:
        image_character = character((200,200), 1, 0.25)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        count: int
            Number to be displayed.
        percent: float
            Percentage of completion.
    Returns: np.ndarray
        Image of pre-roll screen.
    """
    return (text(shape, f"{count:02d}", position=(0.5 * shape[0], 0.57 * shape[1]), font_size=font_size) + vortex(shape, 0.25, percent * np.pi)) / 2


def airy_fn(xx_yy: tuple[np.ndarray, np.ndarray], center: tuple[float, float], radius: float, height: float):
    """
    2d airy function (used for fitting).

    Example:
        image_airy = Airy_fn(TODO: example)

    Parameters:
        xx_yy: tuple[np.ndarray, np.ndarray]
            X Y coordinate arrays.
        center: tuple[float, float]
            Gaussian center coordinates.
        radius: float
            Radius.
        A: float
            Height of the airy function.

    Returns: np.ndarray
        Image of the airy function.
    """

    xx, yy = xx_yy
    x0, y0 = center
    r = radius * np.hypot((xx - x0), (yy - y0))
    return height * np.where(r, (2 * special.jv(1, r) / r) ** 2, 1)


def airy(shape: tuple[int, int], center: tuple[float, float] = (0, 0), radius: float = 1, height: float = 1):
    """
    Create a 2d Airy disk.

    Example:
        image_airy = Airy2d(TODO: example)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        center: tuple[float, float]
            Gaussian center coordinates.
        radius: float
            Radius.
        height: float
            Height of the airy function.

    Returns: np.ndarray
        Image of the Airy disk.
    """
    xx, yy = generate_coordinates(shape, (0, 0), cartesian=True)
    return airy_fn((xx, yy), center, radius, height)


def linear2d_fn(xx_yy: tuple[np.ndarray, np.ndarray], a: float, b: float, c: float):
    """
    2d plane function (used for fitting).

    Example:
        image_airy = Linear2d_fn(TODO: example)

    Parameters:
        xx_yy: tuple[np.ndarray, np.ndarray]
            X Y coordinate arrays.
        a: float
            Constant.
        b: float
            Constant.
        c: float
            intercept constant.

    Returns: np.ndarray
        Image of the plane.
    """

    xx, yy = xx_yy
    return a * xx + b * yy + c


def linear2d(shape: tuple[int, int], a: float, b: float, c: float):
    """
    Create a 2d plane.

    Example:
        image_plane = Linear2d(TODO: example)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        a: float
            Constant.
        b: float
            Constant.
        c: float
            intercept constant.

    Returns: np.ndarray
        Image of the plane.
    """
    xx, yy = generate_coordinates(shape, (0, 0), cartesian=True)
    return linear2d_fn((xx, yy), a, b, c)


def least_squares_fit(y_data: np.ndarray, model, guess_prms=None, bounds=(-np.inf, np.inf), x_coord: np.ndarray | None = None, mask: np.ndarray | None = None):
    """
    Fit a function to an array of values.

    Example:
        def fit_quadratic_fn(x, a, b, c):
            return a*x*x + b*x + c

        a, b, c = 0.25, -1, 2.5
        data = fit_quadratic_fn(np.linspace(0, 10, 21), a, b, c)

        least_squares_fit(data, fit_quadratic_fn, guess_prms=(0.26, -1.1, 2.6), x_coord=np.linspace(0, 10, 21))

    Parameters:
        y_data: np.ndarray
            Array of values to fit.
        model: function(x:np.ndarray, ...)
            Fit function.
            The first parameter must be np.ndarray.
            The rest of the parameters must not be tuples.
        guess_prms: tuple[...]
            guess parameters.
        bounds: tuple[list[...],list[...]]
            guess parameters. a tuple of two lists one upper and the other lower bounds.
        x_coord: np.ndarray
            x coordinates of the data set.
        mask: np.ndarray
            mask of value to fit to.

    Returns: tuple[tuple[...], np.ndarray]
        fit parameters
        goodness of fit
    """

    if mask is None:
        mask = np.isnan(y_data)

    if x_coord is None:
        x_coord = np.linspace(0, y_data.shape[0], y_data.shape[0], endpoint=False)

    y_fit = np.ma.masked_array(y_data, mask=mask)
    x_fit = np.ma.masked_array(x_coord, mask=mask)

    return curve_fit(model, x_fit, y_fit, p0=guess_prms, bounds=bounds)


def least_squares_fit_2d(z_data, model, guess_prms=None, bounds=(-np.inf, np.inf), xy_coords=None, mask=None):
    """
    Fit a 2d function to an 2d array of values.

    Example:
        linear2d_data = Linear2d((200, 200), 3, 2, 1)
        (tip, tilt, piston), __ = least_squares_fit_2d(linear2d_data, Linear2d_fn)

        gauss2d_data = Gauss2d((200, 200), offset=0, height=1, width=(3, 3), center=(100, 100), tilt=0)
        def fit_Gauss2d_fn(xx_yy, center_x, center_y, offset, height, width_x, width_y):
            return Gauss2d_fn(xx_yy, (center_x, center_y), offset, height, (width_x, width_y))
        (center_x, center_y, offset, height, width_x, width_y), _ = least_squares_fit_2d(gauss2d_data, fit_Gauss2d_fn, guess_prms=(101,101, 0, 1.1, 3.3, 3))
        print(center_x, center_y, offset, height, width_x, width_y)

    Parameters:
        z_data: np.ndarray
            2d Array of values to fit.
        model: function(xx_yy:tuple[np.ndarray, np.ndarray], ...)
            Fit function.
            The first parameter must be tuple[np.ndarray, np.ndarray]
            The rest of the parameters must not be tuples.
        guess_prms: tuple[...]
            guess parameters.
        bounds: tuple[list[...],list[...]]
            guess parameters. a tuple of two lists one upper and the other lower bounds.
        xy_coords: tuple[np.ndarray, np.ndarray]
            xy coordinates of the data set.
        mask: np.ndarray
            mask of value to fit to.

    Returns: tuple[tuple[...], np.ndarray]
        fit parameters
        goodness of fit
    """

    if mask is None:
        mask = np.isnan(z_data)

    if xy_coords is None:
        x_coord, y_coord = generate_coordinates(z_data.shape, cartesian=True)
    else:
        x_coord, y_coord = xy_coords

    z_masked = np.ma.masked_array(z_data, mask=mask)
    x_masked = np.ma.masked_array(x_coord, mask=mask)
    y_masked = np.ma.masked_array(y_coord, mask=mask)

    xy_fit = np.vstack((x_masked.compressed().ravel(), y_masked.compressed().ravel()))
    z_fit = z_masked.compressed()

    return curve_fit(model, xy_fit, z_fit, p0=guess_prms, bounds=bounds)


def psf_to_otf(psf: np.ndarray, axes=None) -> NDArray[np.complex64]:
    """
    Calculate the Optical Transfer Function (OTF) from a PSF.

    Example:
        otf = PSF_to_OTF(psf_image)

    Parameters:
        psf: np.ndarray
            PSF image.

    Returns: np.ndarray
        Complex OTF image.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf, axes=axes), axes=axes), axes=axes)


def h_sort(xy: np.ndarray, ascend: bool = True) -> np.ndarray:
    """
    Sort x,y coordinate array based on y value.

    Example:
        xy = h_sort(TODO: example)

    Parameters:
        xy: np.ndarray
            Numpy array with (n,2) elements.
        ascend: bool
            Sorting order.

    Returns: np.ndarray
        Array sorted with the x element.
    """
    if ascend:
        return np.argsort((xy[:, 0]))
    else:
        return np.flip(np.argsort((xy[:, 0])))


def v_sort(xy: np.ndarray, ascend: bool = True) -> np.ndarray:
    """
    Sort x,y coordinate array based on y value.

    Example:
        xy = v_sort(TODO: example)

    Parameters:
        xy: np.ndarray
            Numpy array with (n,2) elements.
        ascend: bool
            Sorting order.

    Returns: np.ndarray
        Array sorted with the y element.
    """
    if ascend:
        return np.argsort((xy[:, 1]))
    else:
        return np.flip(np.argsort((xy[:, 1])))


def xy_sort(xy: np.ndarray, shape: tuple[int, int], h_ascend: bool = True, v_ascend: bool = True) -> np.ndarray:
    """
    Sort x,y coordinate array based on x and y value.

    Example:
        xy = xy_sort(TODO: example)

    Parameters:
        xy: np.ndarray
            Numpy array with (n,2) elements.
        shape: tuple[int, int]
            The arrangement of coordinates.
        h_ascend: bool
            Sorting order.
        v_ascend: bool
            Sorting order.

    Returns: np.ndarray
        Array sorted with the x and y element.
    """
    indices = []
    h_sorted_indices = np.reshape(h_sort(xy, h_ascend), shape)
    for h_sorted_index in h_sorted_indices:
        v_sorted_indices = v_sort(xy[h_sorted_index], v_ascend)
        indices.append(h_sorted_index[v_sorted_indices])
    return np.array(indices).flatten()


def dotf_registration_mask(shape: tuple[int, int], size: float = 0.5, center: tuple[float, float] = (0.5, 0.5)):
    """
    DOTF Registration pattern mask.

    Example:
        mask = dotf_registration_mask(TODO: example)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        size: float
            Size of the mask (normalized).
        center: tuple[float, float]
            Center of the mask (normalized).

    Returns: np.ndarray
        Mask for the DOTF registration pattern.
    """

    _width, _height = shape
    _size_px = (size * _width, size * _height)
    center_px = (center[0] * _width, center[1] * _height)
    return box(shape, _size_px, center_px)


def detect_registration_pattern(image: np.ndarray, mask: np.ndarray, num_peaks: tuple[int, int]):
    """
    Detect registration patterns.

    Example:
        mask = detect_registration_pattern(TODO: example)

    Parameters:
        image: np.ndarray
            Image.
        mask: np.ndarray
            Mask.
        num_peaks: int
            Number of peaks.

    Returns:
        peak_mask: np.ndarray
            Peaks mask.
        position_px: np.ndarray
            Numpy array with (n,2) elements.
    """
    peak_indices = peak_local_max(image * mask, num_peaks=num_peaks[0] * num_peaks[1], min_distance=5, num_peaks_per_label=1)
    peak_mask = np.zeros(image.shape, dtype=bool)
    peak_mask[tuple(peak_indices.T)] = True
    peak_dilation_element = morphology.disk(3)
    peak_mask = morphology.dilation(peak_mask, peak_dilation_element)

    peak_label = label(peak_mask)
    properties = regionprops(peak_label, image)
    positions = np.array([_property.centroid_weighted for _property in properties])
    position_px = positions[xy_sort(positions, num_peaks)]

    return ~peak_mask, position_px


def dotf_to_wavefront(dotf: NDArray[np.complex64], locations: NDArray[np.float64]):
    """
    Convert dotf to a dm command using pixel

    Parameters:
        dotf: NDArray[np.complex64]
            DOTF image.
        locations: NDArray[np.float64] (2, n_acts)
            Actuator locations.

    Returns: nd.ndarray (n_acts)
        Command

    """
    grid_x, grid_y = np.arange(dotf.shape[0]), np.arange(dotf.shape[1])
    real_surface_fn, imag_surface_fn = RegularGridInterpolator((grid_x, grid_y), dotf.real), RegularGridInterpolator((grid_x, grid_y), dotf.imag)
    wavefront = np.zeros(locations.shape[0], dtype=np.complex64)
    for _index, _location in enumerate(locations):
        wavefront[_index] = real_surface_fn((_location[0], _location[1])) + imag_surface_fn((_location[0], _location[1])) * 1j
    return wavefront


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compound_dotf_to_wavefront(shape: tuple[int, int], dotfs: list[NDArray[np.complex64]], locations: list[NDArray[np.float64]]):
    """
    Convert 4 dotf maps to commands and combine them to one command

    Parameters:
        shape: tuple[int, int]
            Shape of command.
        dotfs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The DOTF images.
        locations: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Actuator positions in each DOTF image.

    Returns: nd.ndarray
        Command
    """

    wavefronts = np.full((len(dotfs), *shape), 0, dtype=np.complex64)
    dsk_radius = shape[0] / 2 + 0.5
    box_center = dsk_center = (-shape[0] / 2, -shape[1] / 2)
    box_shape = (dsk_radius / np.sqrt(2), dsk_radius / np.sqrt(2))

    dsk_mask = disk(shape, dsk_radius, dsk_center)  # circular binary mask

    xx, _ = generate_coordinates(shape, cartesian=True)
    prb_mask = 1 - smoothstep(25, 30, xx)  # step grayscale mask to remove probe

    box_mask = box(shape, box_shape, box_center)

    for index, (wavefront, dotf, all_act_loc_px) in enumerate(zip(wavefronts, dotfs, locations)):
        wavefront[dsk_mask] = dotf_to_wavefront(dotf, all_act_loc_px)
        wavefront_abs = np.abs(wavefront)
        wavefront_phs = unwrap_phase(np.angle(wavefront))
        avg_phs = np.nanmean(wavefront_phs[box_mask])
        wavefront_phs = wavefront_phs - avg_phs
        wavefronts[index] = wavefront_abs * np.exp(wavefront_phs * 1j)
        wavefronts[index] = wavefronts[index] * np.rot90(prb_mask, index)

    compound_wavefront = np.ma.array(np.zeros(shape, dtype=np.complex64), mask=~dsk_mask)
    compound_wavefront.data[:] = np.nanmean(wavefronts, axis=0)

    return np.ma.copy(compound_wavefront)


def timestamp_string(timestamp: float | None = None, frmt="%H:%M:%S", ms: None | int = 3, separator=".") -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now().timestamp()
    stamp_seconds = datetime.datetime.fromtimestamp(timestamp).strftime(frmt)
    if ms is None:
        return stamp_seconds
    else:
        return f"{stamp_seconds}{separator}{int((timestamp % 1) * (10**ms)):03d}"


def overlay_info_string(timestamp: float | None = None, exp_time_ms: int = 0, gain: float = 0, rate: float = 0, temp: float = 0, br: tuple[int, int] = (0, 0), tl: tuple[int, int] = (0, 0), frame: int = 0, event: int = 0):
    return f"Time     : {timestamp_string(timestamp)}\nExp Time : {exp_time_ms}\nGain     : {gain}\nRate     : {rate}\nTemp     : {temp}\nROI\n br      : [{br[0]},{br[1]}]\n tl      : [{tl[0]},{tl[1]}]\nFrame    : {frame}\nEvent    : {event}\n"


def RadialAverage(image, center=(0, 0)):
    width, height = image.shape

    if not (center):
        center = tuple(np.array(image.shape) // 2)

    xx, yy = np.ogrid[0:width, 0:height]
    rr = np.hypot(xx - center[0], yy - center[1]).astype(np.uint64)

    r = np.arange(0, np.max(rr))
    avg = ndimage.mean(image, rr, index=r)

    return r, avg


def describe_array(arr: np.ndarray, vals: bool = False) -> str:
    return f"Shape: {arr.shape}\ndtype: {arr.dtype}\nMin:   {np.nanmin(arr)}\nMax:   {np.nanmax(arr)}" + (f"\n{arr}\n" if vals else "\n")


def invert_2x2_arrays(a: float | np.ndarray, b: float | np.ndarray, c: float | np.ndarray, d: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    Δ = a * d - b * c
    p, q = d / Δ, -b / Δ
    r, s = -c / Δ, a / Δ
    return p, q, r, s
