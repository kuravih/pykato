import unittest
from pykato.plotfunction.gridspec_layout import GridSpec_Layout
from pykato.plotfunction.preset import Imshow_Colorbar_Preset, Complex_Imshow_2Colorbars_Preset, ImageGrid_Preset, ImageGrid_Colorbar_Preset, Complex_ImageGrid_2Colorbars_Preset
from pykato.function import polka, vortex, checkers, sinusoid, Gauss2d
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


class TestGridSpecLayout(unittest.TestCase):
    def test_Simple_Layout(self):
        simple_layout_figure = GridSpec_Layout(1, 1)
        self.assertIsInstance(simple_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        (imshow_ax,) = simple_layout_figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        simple_layout_figure.savefig("tests/output/test_Simple_Layout.png")

    def test_Image_Layout(self):
        image_layout_figure = GridSpec_Layout(1, 1, aspect_ratios=(1,))
        self.assertIsInstance(image_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        (imshow_ax,) = image_layout_figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        image_layout_figure.savefig("tests/output/test_Image_Layout.png")

    def test_Images_Layout(self):
        fourimages_layout_figure = GridSpec_Layout(1, 4, aspect_ratios=(1, 1, 1, 1), width_ratios=(1, 1, 1, 1), wspace=0.05)
        self.assertIsInstance(fourimages_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax = fourimages_layout_figure.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "imshow1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow2_ax, Axes, "imshow2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow3_ax, Axes, "imshow3_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow4_ax, Axes, "imshow4_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        fourimages_layout_figure.savefig("tests/output/test_Images_Layout.png")

    def test_Image_Colorbar_Layout(self):
        image_colorbar_layout_figure = GridSpec_Layout(1, 2, aspect_ratios=(1, 0.05), width_ratios=(1, 0.05), wspace=0.05)
        self.assertIsInstance(image_colorbar_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow_ax, colorbar_ax = image_colorbar_layout_figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar_ax, Axes, "colorbar_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        image_colorbar_layout_figure.savefig("tests/output/test_Image_Colorbar_Layout.png")

    def test_Images_Colorbar_Layout(self):
        fourimages_colorbar_layout_figure = GridSpec_Layout(1, 5, aspect_ratios=(1, 1, 1, 1, 0.05), width_ratios=(1, 1, 1, 1, 0.05), wspace=0.05)
        self.assertIsInstance(fourimages_colorbar_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, colorbar_ax = fourimages_colorbar_layout_figure.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "imshow1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow2_ax, Axes, "imshow2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow3_ax, Axes, "imshow3_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow4_ax, Axes, "imshow4_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar_ax, Axes, "colorbar_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        fourimages_colorbar_layout_figure.savefig("tests/output/test_Images_Colorbar_Layout.png")

    def test_Image_TwoColorbars_Layout(self):
        image_twocolorbars_layout_figure = GridSpec_Layout(1, 3, aspect_ratios=(1, 0.05, 0.05), width_ratios=(1, 0.05, 0.05), wspace=0.05)
        self.assertIsInstance(image_twocolorbars_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow_ax, colorbar1_ax, colorbar2_ax = image_twocolorbars_layout_figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar1_ax, Axes, "colorbar1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar2_ax, Axes, "colorbar2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        image_twocolorbars_layout_figure.savefig("tests/output/test_Image_TwoColorbars_Layout.png")

    def test_Images_TwoColorbars_Layout(self):
        fourimages_twocolorbars_layout_figure = GridSpec_Layout(1, 6, aspect_ratios=(1, 1, 1, 1, 0.05, 0.05), width_ratios=(1, 1, 1, 1, 0.05, 0.05), wspace=0.05)
        self.assertIsInstance(fourimages_twocolorbars_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, colorbar1_ax, colorbar2_ax = fourimages_twocolorbars_layout_figure.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "imshow1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow2_ax, Axes, "imshow2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow3_ax, Axes, "imshow3_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow4_ax, Axes, "imshow4_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar1_ax, Axes, "colorbar1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar2_ax, Axes, "colorbar2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        fourimages_twocolorbars_layout_figure.savefig("tests/output/test_Images_TwoColorbars_Layout.png")


class BaseTestPrest(unittest.TestCase):
    def setUp(self):
        polka_test_image = polka((100, 100), 4, (20, 20), (-10, -10))
        checkers_test_image = checkers((100, 100), (25, 25), (-12.5, 12.5))
        sinusoid_test_image = sinusoid((100, 100), 20, 270, 45)
        gauss_test_image = Gauss2d((100, 100), 0, 1, (20, 20), (50, 50))
        vortex_test_image = np.pi * (vortex((100, 100), 2) * 2 - 1)
        polka_vortex_complex_test_image = polka_test_image * np.exp(1j * vortex_test_image)
        checkers_vortex_complex_test_image = checkers_test_image * np.exp(1j * vortex_test_image)
        sinusoid_vortex_complex_test_image = sinusoid_test_image * np.exp(1j * vortex_test_image)
        gauss_vortex_complex_test_image = gauss_test_image * np.exp(1j * vortex_test_image)
        self.data = {"polka_test_image": polka_test_image, "checkers_test_image": checkers_test_image, "sinusoid_test_image": sinusoid_test_image, "gauss_test_image": gauss_test_image, "vortex_test_image": vortex_test_image, "polka_vortex_complex_test_image": polka_vortex_complex_test_image, "checkers_vortex_complex_test_image": checkers_vortex_complex_test_image, "sinusoid_vortex_complex_test_image": sinusoid_vortex_complex_test_image, "gauss_vortex_complex_test_image": gauss_vortex_complex_test_image}


class TestPreset(BaseTestPrest):
    def test_Imshow_Colorbar_Preset(self):
        imshow_colorbar_preset = Imshow_Colorbar_Preset(self.data["polka_test_image"])
        imshow_colorbar_preset.get_image().set_clim(0, 1)

        self.assertIsInstance(imshow_colorbar_preset, Figure, "Figure not created by preset.Imshow_Colorbar_Preset")

        imshow_ax, colorbar_ax = imshow_colorbar_preset.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by preset.Imshow_Colorbar_Preset")
        self.assertIsInstance(colorbar_ax, Axes, "Colorbar axis not available in Figure created by preset.Imshow_Colorbar_Preset")

        imshow_colorbar_preset.savefig("tests/output/preset_Imshow_Colorbar_Preset.png")

    def test_Complex_Imshow_2Colorbars_Preset(self):
        complex_imshow_2colorbars_preset = Complex_Imshow_2Colorbars_Preset(self.data["polka_vortex_complex_test_image"])

        self.assertIsInstance(complex_imshow_2colorbars_preset, Figure, "Figure not created by preset.Complex_Imshow_2Colorbars_Preset")

        imshow_ax, mod_colorbar_ax, arg_colorbar_ax = complex_imshow_2colorbars_preset.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_Imshow_2Colorbars_Preset")
        self.assertIsInstance(mod_colorbar_ax, Axes, "Colorbar axis not available in Figure created by preset.Complex_Imshow_2Colorbars_Preset")
        self.assertIsInstance(arg_colorbar_ax, Axes, "Colorbar axis not available in Figure created by preset.Complex_Imshow_2Colorbars_Preset")

        complex_imshow_2colorbars_preset.savefig("tests/output/preset_Complex_Imshow_2Colorbars_Preset.png")

    def test_ImageGrid_Preset(self):
        imagegrid_preset = ImageGrid_Preset(tuple(self.data[key] for key in ("polka_test_image", "checkers_test_image", "gauss_test_image", "sinusoid_test_image")))

        self.assertIsInstance(imagegrid_preset, Figure, "Figure not created by preset.ImageGrid_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, _, _, _, _ = imagegrid_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Preset")

        imagegrid_preset.savefig("tests/output/preset_ImageGrid_Preset.png")

    def test_ImageGrid_Colorbar_Preset(self):
        imagegrid_colorbar_preset = ImageGrid_Colorbar_Preset(tuple(self.data[key] for key in ("polka_test_image", "checkers_test_image", "gauss_test_image", "sinusoid_test_image")))

        self.assertIsInstance(imagegrid_colorbar_preset, Figure, "Figure not created by preset.ImageGrid_Colorbar_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, _, _, _, _ = imagegrid_colorbar_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Colorbar_Preset")

        imagegrid_colorbar_preset.savefig("tests/output/preset_ImageGrid_Colorbar_Preset.png")

    def test_Complex_ImageGrid_2Colorbars_Preset(self):
        complex_imagegrid_2colorbars_preset = Complex_ImageGrid_2Colorbars_Preset(tuple(self.data[key] for key in ("polka_vortex_complex_test_image", "checkers_vortex_complex_test_image", "sinusoid_vortex_complex_test_image", "gauss_vortex_complex_test_image")))

        self.assertIsInstance(complex_imagegrid_2colorbars_preset, Figure, "Figure not created by preset.Complex_ImageGrid_2Colorbars_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, _, _ = complex_imagegrid_2colorbars_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_2Colorbars_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_2Colorbars_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_2Colorbars_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_2Colorbars_Preset")

        complex_imagegrid_2colorbars_preset.savefig("tests/output/preset_Complex_ImageGrid_2Colorbars_Preset.png")
