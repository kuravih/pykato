import numpy as np
from numpy.typing import NDArray
from .function import generate_coordinates
from skimage.restoration import unwrap_phase
import zern.zern_core as zern  # import the main library


def wavefront_to_phase_delay(wavefront: NDArray[np.complex64], λ: float = 632.992) -> NDArray[np.float64]:
    """
    Calculate the phase delay in nanometers from the a complex wavefront.

    Parameters:
        wavefront: NDArray[np.complex64]
            Wavefront
        λ: float = 632.992
            Wavelength

    Returns: NDArray[np.float64]:
        Phase delay of the wavefront
    """
    phase = unwrap_phase(np.angle(wavefront) + np.pi)
    phase_delay_nm = λ * phase / (2 * np.pi)
    # --- piston, tip, tilt, defocus removal --------------------------------------------------------------------------
    rr, θθ = generate_coordinates(phase_delay_nm.shape, offset=(-phase_delay_nm.shape[0] / 2 + 0.5, -phase_delay_nm.shape[1] / 2 + 0.5), polar=True)
    rho = np.ma.array(rr, mask=phase_delay_nm.mask)
    theta = np.ma.array(θθ, mask=phase_delay_nm.mask)
    zernobj = zern.Zernike(mask=phase_delay_nm.mask)
    n, m = [0, 1, 1, 2], [0, -1, 1, 0]  # piston, tip, tilt, defocus
    zernobj.create_model_matrix(rho.compressed(), theta.compressed(), n, m, mode="Jacobi", normalize_noll=True)
    piston, tip, tilt, defocus = zernobj.decompose(phase_delay_nm.compressed())
    phase_delay_nm_fit = zernobj.get_zernike((piston, tip, tilt, defocus))
    phase_delay_nm_residual = phase_delay_nm - phase_delay_nm_fit
    # --- piston, tip, tilt, defocus removal --------------------------------------------------------------------------
    return phase_delay_nm_residual
