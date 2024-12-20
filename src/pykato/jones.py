from typing import Tuple
import numpy as np


def su2(a: complex, b: complex) -> Tuple[Tuple[complex, complex], Tuple[complex, complex]]:
    """
    Generate an su2 matrix.

    Examples:
        mat = su2(1,1)

    Parameter:
        a: complex
            complex value.
        b: complex
            complex value.

    Returns: 2-(2-tuple)
        su2 matrix.
    """
    return (a, -np.conjugate(b)), (b, np.conjugate(a))


def calc_terms(θ: float, φ: float, η: float) -> Tuple[complex, complex]:
    """
    Calculate a and b terms for the retarder matrix. See https://en.wikipedia.org/wiki/Jones_calculus#Phase_retarders.

    Example:
        a, b = calc_terms(np.pi, np.pi, 0)

    Parameters:
        θ: float
            Angle of the fast axis of the retarder to the horizontal.
        φ: float
            Ellipticity of the retarder.
        eta: float
            Phase retardance.

    Returns: 2-tuple
        a, b for the su2 matrix.
    """

    return np.cos(η / 2) - 1j * np.sin(η / 2) * np.cos(2 * θ), np.sin(η / 2) * np.sin(φ) * np.sin(2 * θ) - 1j * np.sin(η / 2) * np.cos(φ) * np.sin(2 * θ)


def retarder(θ: float, η: float, φ: float) -> Tuple[Tuple[complex, complex], Tuple[complex, complex]]:
    """
    Jones matrix for a variable retarder at an arbitrary angle θ from the horizontal.

    Example:
        r = retarder(np.pi, np.pi, 0)

    Parameters:
        θ: float
            Angle of the fast axis of the retarder from the horizontal.
        φ: float
            Ellipticity of the retarder.
        η: float
            Phase retardance.

    Returns: 2-(2-tuple)
        Jones matrix of retarder.
    """

    return su2(*calc_terms(θ, φ, η))


def linear_polarizer(θ: float) -> Tuple[Tuple[complex, complex], Tuple[complex, complex]]:
    """
    Jones matrix for a linear polarizer at an arbitrary angle θ from the horizontal.

    Example:
        lp = linear_polarizer(np.pi/2)

    Parameters:
        θ: float
            Angle of the linear to the horizontal.

    Returns: 2-(2-tuple)
        Jones matrix for a linear polarizer.
    """
    return (np.cos(θ) ** 2, np.cos(θ) * np.sin(θ)), (np.cos(θ) * np.sin(θ), np.sin(θ) ** 2)


def half_wave_plate(θ: float) -> Tuple[Tuple[complex, complex], Tuple[complex, complex]]:
    """
    Jones matrix for a half wave plate with the fast axis at an arbitrary angle θ from the horizontal.

    Example:
        hwp = half_wave_plate(np.pi/4)

    Parameters:
        θ: float
            Angle of the linear to the horizontal.

    Returns: 2-(2-tuple)
        Jones matrix for a half wave plate.
    """
    return retarder(θ, np.pi, 0)


def quarter_wave_plate(θ: float) -> Tuple[Tuple[complex, complex], Tuple[complex, complex]]:
    """
    Jones matrix for a quarter wave plate with the fast axis at an arbitrary angle θ from the horizontal.

    Example:
        qwp = quarter_wave_plate(np.pi/4)

    Parameters:
        θ: float
            Angle of the linear to the horizontal.

    Returns: 2-(2-tuple)
        Jones matrix for a quarter wave plate.
    """
    return retarder(θ, np.pi / 2, 0)


def magnitude(x: complex, y: complex) -> float:
    """
    Magnitude of a jones vector.

    Example:
        mag = magnitude(1,0)

    Parameters:
        x: complex
            x component of light
        y: complex
            y component of light

    Returns: float
        Intensity of light
    """
    return np.sqrt(np.conjugate(x) * x + np.conjugate(y) * y)


def lp(θ: float) -> Tuple[complex, complex]:
    """
    Jones vector for linear polarized light at an arbitrary arbitrary angle θ from the horizontal.

    Example:
        input = lp(np.pi/4)

    Parameters:
        θ: float
            Angle from the horizontal.

    Returns: Tuple[complex, complex]
        Jones vector of light.
    """
    x, y = np.cos(θ), np.sin(θ)
    return x / magnitude(x, y), y / magnitude(x, y)


def hlp() -> Tuple[complex, complex]:
    """
    Jones vector for horizontally linear polarized light.

    Example:
        input = hlp()

    Returns: Tuple[complex, complex]
        Jones vector of light.
    """
    return 1, 0


def vlp() -> Tuple[complex, complex]:
    """
    Jones vector for vertically linear polarized light.

    Example:
        input = vlp()

    Returns: Tuple[complex, complex]
        Jones vector of light.
    """
    return 0, 1


def rcp() -> Tuple[complex, complex]:
    """
    Jones vector for right hand circularly polarized light.

    Example:
        input = rcp()

    Returns: Tuple[complex, complex]
        Jones vector of light.
    """
    return 1, -1j


def lcp() -> Tuple[complex, complex]:
    """
    Jones vector for left hand circularly polarized light.

    Example:
        input = lcp()

    Returns: Tuple[complex, complex]
        Jones vector of light.
    """
    return 1, +1j
