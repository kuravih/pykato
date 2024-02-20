import h5py as h5
import numpy as np


def h5_tree(val, pre=""):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5._hl.group.Group:
                print(f"{pre}└──{key}")
                h5_tree(val, f"{pre}    ")
            else:
                print(f"{pre}└── {key} [{len(val)}]")
        else:
            if type(val) == h5._hl.group.Group:
                print(f"{pre}├── {key}")
                h5_tree(val, f"{pre}│   ")
            else:
                print(f"{pre}├── {key} [{len(val)}]")


class Measurement(h5.File):
    """
    Usage
    -----
        part_01 = FourD.Measurement('1.4D')
        part_01.Intensity : for intensity
        part_01.Surface : for surface in nanometers
        part_01.Wavelength : for wavelength
        part_01.CalculatedZernikeTerms : for calculated zernike terms
    """

    def __init__(self, _path):
        super(Measurement, self).__init__(_path, "r")

    @property
    def Intensity(self):
        return np.ma.masked_invalid(self["Measurement"]["Intensity"]["Data"])

    @property
    def Wavelength(self):
        return self["Measurement"].attrs["WavelengthInNanometers"]

    @property
    def Surface(self):
        return np.ma.masked_invalid(self["Measurement"]["SurfaceInWaves"]["Data"] * self.Wavelength)

    @property
    def CalculatedZernikeTerms(self):
        return np.array(self["Measurement"]["CalculatedZernikeTerms"])
