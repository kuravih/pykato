import unittest
from pykato.plotfunction.gridspec_layout import GridSpec_Layout
from pykato.plotfunction.preset import Imshow_Colorbar_Preset, Complex_Imshow_TwoColorbars_Preset, ImageGrid_Preset, ImageGrid_Colorbar_Preset, Complex_ImageGrid_TwoColorbars_Preset, Imshow_Colorbar_Imshow_Colorbar_Preset, Histogram_Colorbar_Preset, Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset, Imshow_Colorbar_Imshow_Colorbar_Plot_Preset
from pykato.function import polka, vortex, checkers, sinusoid, gauss2d
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
        four_images_layout_figure = GridSpec_Layout(1, 4, aspect_ratios=(1, 1, 1, 1), width_ratios=(1, 1, 1, 1), wspace=0.05)
        self.assertIsInstance(four_images_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax = four_images_layout_figure.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "imshow1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow2_ax, Axes, "imshow2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow3_ax, Axes, "imshow3_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow4_ax, Axes, "imshow4_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        four_images_layout_figure.savefig("tests/output/test_Images_Layout.png")

    def test_Image_Colorbar_Layout(self):
        image_colorbar_layout_figure = GridSpec_Layout(1, 2, aspect_ratios=(1, 0.05), width_ratios=(1, 0.05), wspace=0.05)
        self.assertIsInstance(image_colorbar_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow_ax, colorbar_ax = image_colorbar_layout_figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar_ax, Axes, "colorbar_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        image_colorbar_layout_figure.savefig("tests/output/test_Image_Colorbar_Layout.png")

    def test_Images_Colorbar_Layout(self):
        four_images_colorbar_layout_figure = GridSpec_Layout(1, 5, aspect_ratios=(1, 1, 1, 1, 0.05), width_ratios=(1, 1, 1, 1, 0.05), wspace=0.05)
        self.assertIsInstance(four_images_colorbar_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, colorbar_ax = four_images_colorbar_layout_figure.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "imshow1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow2_ax, Axes, "imshow2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow3_ax, Axes, "imshow3_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow4_ax, Axes, "imshow4_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar_ax, Axes, "colorbar_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        four_images_colorbar_layout_figure.savefig("tests/output/test_Images_Colorbar_Layout.png")

    def test_Image_TwoColorbars_Layout(self):
        image_two_colorbars_layout_figure = GridSpec_Layout(1, 3, aspect_ratios=(1, 0.05, 0.05), width_ratios=(1, 0.05, 0.05), wspace=0.05)
        self.assertIsInstance(image_two_colorbars_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow_ax, colorbar1_ax, colorbar2_ax = image_two_colorbars_layout_figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar1_ax, Axes, "colorbar1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar2_ax, Axes, "colorbar2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        image_two_colorbars_layout_figure.savefig("tests/output/test_Image_TwoColorbars_Layout.png")

    def test_Images_TwoColorbars_Layout(self):
        four_images_two_colorbars_layout_figure = GridSpec_Layout(1, 6, aspect_ratios=(1, 1, 1, 1, 0.05, 0.05), width_ratios=(1, 1, 1, 1, 0.05, 0.05), wspace=0.05)
        self.assertIsInstance(four_images_two_colorbars_layout_figure, Figure, "Figure not created by gridspec_layout.GridSpec_Layout")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, colorbar1_ax, colorbar2_ax = four_images_two_colorbars_layout_figure.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "imshow1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow2_ax, Axes, "imshow2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow3_ax, Axes, "imshow3_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(imshow4_ax, Axes, "imshow4_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar1_ax, Axes, "colorbar1_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")
        self.assertIsInstance(colorbar2_ax, Axes, "colorbar2_ax Axis not available in Figure created by gridspec_layout.GridSpec_Layout")

        four_images_two_colorbars_layout_figure.savefig("tests/output/test_Images_TwoColorbars_Layout.png")


class BaseTestPreset(unittest.TestCase):
    def setUp(self):
        polka_test_image = polka((100, 100), 4, (20, 20), (-10, -10))
        checkers_test_image = checkers((100, 100), (25, 25), (-12.5, 12.5))
        sinusoid_test_image = sinusoid((100, 100), 20, 270, 45)
        gauss_test_image = gauss2d((100, 100), 0, 1, (20, 20), (50, 50))
        vortex_test_image = np.pi * (vortex((100, 100), 2) * 2 - 1)
        polka_vortex_complex_test_image = polka_test_image * np.exp(1j * vortex_test_image)
        checkers_vortex_complex_test_image = checkers_test_image * np.exp(1j * vortex_test_image)
        sinusoid_vortex_complex_test_image = sinusoid_test_image * np.exp(1j * vortex_test_image)
        gauss_vortex_complex_test_image = gauss_test_image * np.exp(1j * vortex_test_image)
        self.data = {"polka_test_image": polka_test_image, "checkers_test_image": checkers_test_image, "sinusoid_test_image": sinusoid_test_image, "gauss_test_image": gauss_test_image, "vortex_test_image": vortex_test_image, "polka_vortex_complex_test_image": polka_vortex_complex_test_image, "checkers_vortex_complex_test_image": checkers_vortex_complex_test_image, "sinusoid_vortex_complex_test_image": sinusoid_vortex_complex_test_image, "gauss_vortex_complex_test_image": gauss_vortex_complex_test_image}


class TestPreset(BaseTestPreset):
    def test_Imshow_Colorbar_Preset(self):
        imshow_colorbar_preset = Imshow_Colorbar_Preset(self.data["polka_test_image"])
        imshow_colorbar_preset.get_image().set_clim(0, 1)

        self.assertIsInstance(imshow_colorbar_preset, Figure, "Figure not created by preset.Imshow_Colorbar_Preset")

        imshow_ax, colorbar_ax = imshow_colorbar_preset.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by preset.Imshow_Colorbar_Preset")
        self.assertIsInstance(colorbar_ax, Axes, "Colorbar axis not available in Figure created by preset.Imshow_Colorbar_Preset")

        imshow_colorbar_preset.savefig("tests/output/preset_Imshow_Colorbar_Preset.png")

    def test_Complex_Imshow_TwoColorbars_Preset(self):
        complex_imshow_two_colorbars_preset = Complex_Imshow_TwoColorbars_Preset(self.data["polka_vortex_complex_test_image"])

        self.assertIsInstance(complex_imshow_two_colorbars_preset, Figure, "Figure not created by preset.Complex_Imshow_TwoColorbars_Preset")

        imshow_ax, mod_colorbar_ax, arg_colorbar_ax = complex_imshow_two_colorbars_preset.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_Imshow_TwoColorbars_Preset")
        self.assertIsInstance(mod_colorbar_ax, Axes, "Colorbar axis not available in Figure created by preset.Complex_Imshow_TwoColorbars_Preset")
        self.assertIsInstance(arg_colorbar_ax, Axes, "Colorbar axis not available in Figure created by preset.Complex_Imshow_TwoColorbars_Preset")

        complex_imshow_two_colorbars_preset.savefig("tests/output/preset_Complex_Imshow_2Colorbars_Preset.png")

    def test_ImageGrid_Preset(self):
        image_grid_preset = ImageGrid_Preset(tuple(self.data[key] for key in ("polka_test_image", "checkers_test_image", "gauss_test_image", "sinusoid_test_image")))

        self.assertIsInstance(image_grid_preset, Figure, "Figure not created by preset.ImageGrid_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, _, _, _, _ = image_grid_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Preset")

        image_grid_preset.savefig("tests/output/preset_ImageGrid_Preset.png")

    def test_ImageGrid_Colorbar_Preset(self):
        image_grid_colorbar_preset = ImageGrid_Colorbar_Preset(tuple(self.data[key] for key in ("polka_test_image", "checkers_test_image", "gauss_test_image", "sinusoid_test_image")))

        self.assertIsInstance(image_grid_colorbar_preset, Figure, "Figure not created by preset.ImageGrid_Colorbar_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, _, _, _, _ = image_grid_colorbar_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow axis not available in Figure created by preset.ImageGrid_Colorbar_Preset")

        image_grid_colorbar_preset.savefig("tests/output/preset_ImageGrid_Colorbar_Preset.png")

    def test_Complex_ImageGrid_TwoColorbars_Preset(self):
        complex_image_grid_two_colorbars_preset = Complex_ImageGrid_TwoColorbars_Preset(tuple(self.data[key] for key in ("polka_vortex_complex_test_image", "checkers_vortex_complex_test_image", "sinusoid_vortex_complex_test_image", "gauss_vortex_complex_test_image")))

        self.assertIsInstance(complex_image_grid_two_colorbars_preset, Figure, "Figure not created by preset.Complex_ImageGrid_TwoColorbars_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, _, _ = complex_image_grid_two_colorbars_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")

        complex_image_grid_two_colorbars_preset.savefig("tests/output/preset_Complex_ImageGrid_2Colorbars_Preset.png")

    def test_Complex_ImageGrid_TwoColorbars_Preset_set_data(self):
        complex_image_grid_two_colorbars_preset = Complex_ImageGrid_TwoColorbars_Preset(tuple(self.data[key] for key in ("polka_vortex_complex_test_image", "checkers_vortex_complex_test_image", "sinusoid_vortex_complex_test_image", "gauss_vortex_complex_test_image")))

        self.assertIsInstance(complex_image_grid_two_colorbars_preset, Figure, "Figure not created by preset.Complex_ImageGrid_TwoColorbars_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, _, _ = complex_image_grid_two_colorbars_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")

        complex_image_grid_two_colorbars_preset.get_images()[0].set_data(self.data['polka_vortex_complex_test_image'])
        complex_image_grid_two_colorbars_preset.get_images()[1].set_data(self.data['polka_vortex_complex_test_image'])
        complex_image_grid_two_colorbars_preset.get_images()[2].set_data(self.data['polka_vortex_complex_test_image'])
        complex_image_grid_two_colorbars_preset.get_images()[3].set_data(self.data['polka_vortex_complex_test_image'])

        complex_image_grid_two_colorbars_preset.savefig("tests/output/preset_Complex_ImageGrid_2Colorbars_Preset_set_data.png")

    def test_Imshow_Colorbar_Imshow_Colorbar_Preset(self):
        imshow_colorbar_imshow_colorbar_preset = Imshow_Colorbar_Imshow_Colorbar_Preset(tuple(self.data[key] for key in ("polka_test_image", "checkers_test_image")))

        self.assertIsInstance(imshow_colorbar_imshow_colorbar_preset, Figure, "Figure not created by preset.Imshow_Colorbar_Imshow_Colorbar_Preset")

        imshow1_ax, colorbar1_ax, imshow2_ax, colorbar2_ax = imshow_colorbar_imshow_colorbar_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.Imshow_Colorbar_Imshow_Colorbar_Preset")
        self.assertIsInstance(colorbar1_ax, Axes, "Colorbar axis not available in Figure created by preset.Imshow_Colorbar_Imshow_Colorbar_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.Imshow_Colorbar_Imshow_Colorbar_Preset")
        self.assertIsInstance(colorbar2_ax, Axes, "Colorbar axis not available in Figure created by preset.Imshow_Colorbar_Imshow_Colorbar_Preset")

        imshow_colorbar_imshow_colorbar_preset.savefig("tests/output/preset_Imshow_Colorbar_Imshow_Colorbar_Preset.png")

    def test_Histogram_Colorbar_Preset(self):
        histogram_colorbar_preset = Histogram_Colorbar_Preset(self.data["polka_test_image"] * (2**16 - 1), vmax=2.0**16 - 1, nbins=64, position="bottom")
        histogram_colorbar_preset.set_data(self.data["polka_test_image"])
        histogram_colorbar_preset.set_vlim(0, 1)

        self.assertIsInstance(histogram_colorbar_preset, Figure, "Figure not created by preset.Histogram_Colorbar_Preset")

        histogram_ax, colorbar_ax = histogram_colorbar_preset.get_axes()
        self.assertIsInstance(histogram_ax, Axes, "Histogram axis not available in Figure created by preset.Histogram_Colorbar_Preset")
        self.assertIsInstance(colorbar_ax, Axes, "Colorbar axis not available in Figure created by preset.Histogram_Colorbar_Preset")

        histogram_colorbar_preset.savefig("tests/output/preset_Histogram_Colorbar_Preset.png")

    def test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset(self):
        imshow_colorbar_imshow_colorbar_plot_plot_preset = Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset(tuple(self.data[key] for key in ("polka_test_image", "checkers_test_image")))

        self.assertIsInstance(imshow_colorbar_imshow_colorbar_plot_plot_preset, Figure, "Figure not created by preset.Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")

        imshow1_ax, colorbar1_ax, imshow2_ax, colorbar2_ax, plot_ax_1, plot_ax_2 = imshow_colorbar_imshow_colorbar_plot_plot_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(colorbar1_ax, Axes, "Colorbar axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(colorbar2_ax, Axes, "Colorbar axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(plot_ax_1, Axes, "Plot axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(plot_ax_2, Axes, "Plot axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")

        imshow_colorbar_imshow_colorbar_plot_plot_preset.savefig("tests/output/preset_test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset.png")

    def test_Imshow_Colorbar_Imshow_Colorbar_Plot_Preset(self):
        imshow_colorbar_imshow_colorbar_plot_preset = Imshow_Colorbar_Imshow_Colorbar_Plot_Preset(tuple(self.data[key] for key in ("polka_test_image", "checkers_test_image")))

        self.assertIsInstance(imshow_colorbar_imshow_colorbar_plot_preset, Figure, "Figure not created by preset.Imshow_Colorbar_Plot_Preset")

        imshow1_ax, colorbar1_ax, imshow2_ax, colorbar2_ax, plot_ax = imshow_colorbar_imshow_colorbar_plot_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(colorbar1_ax, Axes, "Colorbar axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(colorbar2_ax, Axes, "Colorbar axis not available in Figure created by preset.test_Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset")
        self.assertIsInstance(plot_ax, Axes, "Plot axis not available in Figure created by preset.test_Imshow_Colorbar_Plot_Preset")

        imshow_colorbar_imshow_colorbar_plot_preset.savefig("tests/output/preset_test_Imshow_Colorbar_Plot_Preset.png")
