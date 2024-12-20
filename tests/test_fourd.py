import unittest
import os
from pykato.fourd import Measurement
from pykato.plotfunction.fourd import Measurement_Preset, Measurement_Zernike_Preset
from pykato.function import disk
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.ndimage import center_of_mass

class TestFourd(unittest.TestCase):
    # pylint: disable=missing-class-docstring

    def test_Measurement(self):
        sample = Measurement(os.getcwd()+"/tests/data/lyot_mirror.4D")
        print(sample.tree())
        self.assertIsInstance(sample, Measurement, "4D file could not be parsed")

    def test_Measurement_Preset(self):

        data = Measurement(os.getcwd()+"/tests/data/lyot_mirror.4D")
        masked_surface = data.surface.copy()
        mask_center = center_of_mass(~masked_surface.mask)
        mask = ~disk(masked_surface.shape, 500, (-mask_center[1], -mask_center[0]))

        measurement_preset = Measurement_Preset(data, mask)

        self.assertIsInstance(measurement_preset, Figure, "Figure not created by preset.Measurement_Zernike_Preset")

        imshow_ax, colorbar_ax = measurement_preset.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by preset.test_Measurement_Preset")
        self.assertIsInstance(colorbar_ax, Axes, "Colorbar axis not available in Figure created by preset.test_Measurement_Preset")

        measurement_preset.savefig("tests/output/preset_test_Measurement_Preset.png")

    def test_Measurement_Zernike_Preset(self):

        data = Measurement(os.getcwd()+"/tests/data/lyot_mirror.4D")
        masked_surface = data.surface.copy()
        mask_center = center_of_mass(~masked_surface.mask)
        mask = ~disk(masked_surface.shape, 500, (-mask_center[1], -mask_center[0]))

        measurement_zernike_preset = Measurement_Zernike_Preset(data, mask)

        self.assertIsInstance(measurement_zernike_preset, Figure, "Figure not created by preset.Measurement_Zernike_Preset")

        imshow1_ax, colorbar1_ax, imshow2_ax, colorbar2_ax, plot_ax = measurement_zernike_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.test_Measurement_Zernike_Preset")
        self.assertIsInstance(colorbar1_ax, Axes, "Colorbar axis not available in Figure created by preset.test_Measurement_Zernike_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.test_Measurement_Zernike_Preset")
        self.assertIsInstance(colorbar2_ax, Axes, "Colorbar axis not available in Figure created by preset.test_Measurement_Zernike_Preset")
        self.assertIsInstance(plot_ax, Axes, "Plot axis not available in Figure created by preset.test_Measurement_Zernike_Preset")

        measurement_zernike_preset.savefig("tests/output/preset_test_Measurement_Zernike_Preset.png")
