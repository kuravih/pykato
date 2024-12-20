import h5py as h5
import numpy as np
from astropy.io import fits


class Measurement(h5.File):
    """
    4D Measurement

    Attributes:
        intensity: np.ma.MaskedArray
            A masked array of intensity data, masking invalid values.
        wavelength: float
            Wavelength value in nanometers from the file's attributes.
        surface: np.ma.MaskedArray
            Surface data in nanometers, calculated from "SurfaceInWaves" and the wavelength.
        calculated_zernike_terms: np.ndarray
            An array of calculated Zernike terms.

    Methods:
        tree(): str
            Returns a formatted string representing the structure of the HDF5 file.

    Usage:
        sample = FourD.Measurement('sample.4D')
        sample.intensity : for intensity
        sample.surface : for surface in nanometers
        sample.wavelength : for wavelength
        sample.calculated_zernike_terms : for calculated zernike terms
    """

    def __init__(self, _path):
        super(Measurement, self).__init__(_path, "r")

    def tree(self) -> str:
        def _tree(item, pre: str = "") -> str:
            out = ""
            keys = list(item.keys())
            for index, key in enumerate(keys):
                # Print the group or dataset name
                if isinstance(item[key], h5.Group):
                    connector = "├─" if index != len(keys) - 1 else "└─"
                    out = out + f"{pre}{connector}[{key}:Group]\n\r"
                    # Recur into the group, with increased indentation
                    out = out + _tree(item[key], pre + ("│  " if index != len(keys) - 1 else "   "))
                elif isinstance(item[key], h5.Dataset):
                    connector = "└─" if index == len(keys) - 1 else "├─"
                    out = out + f"{pre}{connector}[{key}:Dataset]\n\r"
            return out

        return _tree(self)

    @property
    def intensity(self) -> np.ma.MaskedArray:
        return np.ma.masked_invalid(self["Measurement"]["Intensity"]["Data"])

    @property
    def wavelength(self) -> float:
        return self["Measurement"].attrs["WavelengthInNanometers"]

    @property
    def surface(self) -> np.ma.MaskedArray:
        return np.ma.masked_invalid(self["Measurement"]["SurfaceInWaves"]["Data"] * self.wavelength)

    @property
    def calculated_zernike_terms(self) -> np.ndarray:
        return np.array(self["Measurement"]["CalculatedZernikeTerms"])


def convert_to_fits(data: Measurement):
    """Convert 4d Measurement in fo a fits file with Surface, Intensity, Zernike Terms, data units

    Parameter:
        data: Measurement
            4d measurement

    Returns: fits.HDUList
        fits hdu list

    Usage:
        4d_hdu_list = convert_to_fits(Measurement("measurement/oap4_locked.4D"))
    """

    surface_phdu_header = fits.Header()
    surface_phdu_header["DATA"] = "Surface"
    surface_phdu = fits.PrimaryHDU(data.surface.filled(np.nan), header=surface_phdu_header)

    intensity_hdu_header = fits.Header()
    intensity_hdu_header["DATA"] = "Intensity"
    intensity_hdu = fits.ImageHDU(data=data.intensity.filled(np.nan), name="INTENSITY", header=intensity_hdu_header)

    column = fits.Column(name="array", format=f"{data.calculated_zernike_terms.size}E", array=data.calculated_zernike_terms)

    zernike_hdu_header = fits.Header()
    zernike_hdu_header["DATA"] = "Zernike Terms"
    zernike_hdu = fits.BinTableHDU.from_columns([column], name="ZERNIKES", header=zernike_hdu_header)

    hdu_list = fits.HDUList([surface_phdu, intensity_hdu, zernike_hdu])

    return hdu_list
