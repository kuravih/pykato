import numpy as np


def su2(a: complex, b: complex) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
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


def calc_terms(θ: float, φ: float, η: float) -> tuple[complex, complex]:
    """
    Calculate a and b terms for the retarder matrix. See https://en.wikipedia.org/wiki/Jones_calculus#Phase_retarders.

    Example:
        a, b = calc_terms(np.pi, np.pi, 0)

    Parameters:
        θ: float
            Angle of the fast axis of the retarder to the horizontal.
        η: float
            Phase retardance.
        φ: float
            Ellipticity of the retarder.

    Returns: 2-tuple
        a, b for the su2 matrix.
    """

    return np.cos(η / 2) - 1j * np.sin(η / 2) * np.cos(2 * θ), np.sin(η / 2) * np.sin(φ) * np.sin(2 * θ) - 1j * np.sin(η / 2) * np.cos(φ) * np.sin(2 * θ)


def retarder(θ: float, η: float, φ: float) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    """
    Jones matrix for a variable retarder at an arbitrary angle θ from the horizontal.

    Example:
        r = retarder(np.pi, np.pi, 0)

    Parameters:
        θ: float
            Angle of the fast axis of the retarder from the horizontal.
        η: float
            Phase retardance.
        φ: float
            Ellipticity of the retarder.

    Returns: 2-(2-tuple)
        Jones matrix of retarder.
    """

    return su2(*calc_terms(θ, φ, η))


def linear_polarizer(θ: float) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
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


def half_wave_plate(θ: float) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
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


def quarter_wave_plate(θ: float) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
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


def normalize(x: complex, y: complex) -> tuple[complex, complex]:
    """
    Normalize a Jones vector to unit magnitude.

    Example:
        x_norm, y_norm = normalize(2, 2j)

    Parameters:
        x: complex
            x component of light
        y: complex
            y component of light

    Returns: tuple[complex, complex]
        Normalized Jones vector
    """
    mag = magnitude(x, y)
    return (x / mag, y / mag)


def is_polarization_equal(light1: tuple[complex, complex], light2: tuple[complex, complex], tol: float = 1e-10) -> bool:
    """
    Check if two Jones vectors represent the same polarization state.
    Ignores intensity scaling and global phase differences.

    Example:
        equal = is_polarization_equal((1, 1), (1j, 1j))  # True - same 45° linear
        equal = is_polarization_equal((1, 0), (2, 0))    # True - same horizontal
        equal = is_polarization_equal((1, 0), (0, 1))    # False - different

    Parameters:
        light1: tuple[complex, complex]
            First Jones vector (x, y)
        light2: tuple[complex, complex]
            Second Jones vector (x, y)
        tol: float
            Numerical tolerance for comparison

    Returns: bool
        True if same polarization state
    """
    x1, y1 = light1
    x2, y2 = light2

    # Check for zero vectors
    mag1 = magnitude(x1, y1)
    mag2 = magnitude(x2, y2)
    if mag1 < tol or mag2 < tol:
        return False

    # Normalize both vectors to unit magnitude
    x1_norm, y1_norm = normalize(x1, y1)
    x2_norm, y2_norm = normalize(x2, y2)

    # Remove global phase by making the first non-zero component real and positive
    # For light1
    if abs(x1_norm) >= abs(y1_norm):
        phase_factor1 = np.conjugate(x1_norm) / abs(x1_norm) if abs(x1_norm) > tol else 1
    else:
        phase_factor1 = np.conjugate(y1_norm) / abs(y1_norm) if abs(y1_norm) > tol else 1

    x1_final = x1_norm * phase_factor1
    y1_final = y1_norm * phase_factor1

    # For light2
    if abs(x2_norm) >= abs(y2_norm):
        phase_factor2 = np.conjugate(x2_norm) / abs(x2_norm) if abs(x2_norm) > tol else 1
    else:
        phase_factor2 = np.conjugate(y2_norm) / abs(y2_norm) if abs(y2_norm) > tol else 1

    x2_final = x2_norm * phase_factor2
    y2_final = y2_norm * phase_factor2

    # Compare the normalized, phase-adjusted vectors
    diff_x = abs(x1_final - x2_final)
    diff_y = abs(y1_final - y2_final)

    return diff_x < tol and diff_y < tol


def bulk_phase_diff(light1: tuple[complex, complex], light2: tuple[complex, complex]) -> float | None:
    """
    Calculate bulk phase difference between two Jones vectors if they have the same polarization.
    Returns None if they have different polarization states.

    Example:
        phase_diff = bulk_phase_diff((1, 1), (1j, 1j))  # Returns π/2
        phase_diff = bulk_phase_diff((1, 0), (0, 1))    # Returns None

    Parameters:
        light1: tuple[complex, complex]
            First Jones vector (x, y)
        light2: tuple[complex, complex]
            Second Jones vector (x, y)

    Returns: float or None
        Bulk phase difference in radians [-π, π], or None if different polarization
    """
    # First check if they have the same polarization
    if not is_polarization_equal(light1, light2):
        return None

    x1, y1 = light1
    x2, y2 = light2

    # Check for zero vectors
    tol = 1e-10
    mag1 = magnitude(x1, y1)
    mag2 = magnitude(x2, y2)
    if mag1 < tol or mag2 < tol:
        return None

    # Normalize both vectors
    x1_norm, y1_norm = normalize(x1, y1)
    x2_norm, y2_norm = normalize(x2, y2)

    # Calculate global phase of each vector using the dominant component
    # Use the component with larger magnitude for better numerical stability
    if abs(x1_norm) >= abs(y1_norm):
        global_phase1 = np.angle(x1_norm)
    else:
        global_phase1 = np.angle(y1_norm)

    if abs(x2_norm) >= abs(y2_norm):
        global_phase2 = np.angle(x2_norm)
    else:
        global_phase2 = np.angle(y2_norm)

    # Calculate bulk phase difference
    bulk_phase_difference = global_phase2 - global_phase1

    # Wrap to [-π, π] range
    bulk_phase_difference = np.angle(np.exp(1j * bulk_phase_difference))

    return bulk_phase_difference


def lp(θ: float) -> tuple[complex, complex]:
    """
    Jones vector for linear polarized light at an arbitrary arbitrary angle θ from the horizontal.

    Example:
        input = lp(np.pi/4)

    Parameters:
        θ: float
            Angle from the horizontal.

    Returns: tuple[complex, complex]
        Jones vector of light.
    """
    x, y = np.cos(θ), np.sin(θ)
    return x / magnitude(x, y), y / magnitude(x, y)


def hlp() -> tuple[complex, complex]:
    """
    Jones vector for horizontally linear polarized light.

    Example:
        input = hlp()

    Returns: tuple[complex, complex]
        Jones vector of light.
    """
    return 1, 0


def vlp() -> tuple[complex, complex]:
    """
    Jones vector for vertically linear polarized light.

    Example:
        input = vlp()

    Returns: tuple[complex, complex]
        Jones vector of light.
    """
    return 0, 1


def rcp() -> tuple[complex, complex]:
    """
    Jones vector for right hand circularly polarized light.

    Example:
        input = rcp()

    Returns: tuple[complex, complex]
        Jones vector of light.
    """
    return 1, -1j


def lcp() -> tuple[complex, complex]:
    """
    Jones vector for left hand circularly polarized light.

    Example:
        input = lcp()

    Returns: tuple[complex, complex]
        Jones vector of light.
    """
    return 1, +1j
