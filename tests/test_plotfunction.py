import unittest

from pykato.plotfunction.gridspec_layout import GridSpec_Layout
from pykato.plotfunction.preset import Imshow_Preset, Complex_Imshow_Preset, Imshow_Colorbar_Preset, Complex_Imshow_TwoColorbars_Preset, ImageGrid_Preset, ImageGrid_Colorbar_Preset, Complex_ImageGrid_TwoColorbars_Preset, Imshow_Colorbar_Imshow_Colorbar_Preset, Histogram_Colorbar_Preset, Imshow_Colorbar_Imshow_Colorbar_Plot_Plot_Preset, Imshow_Colorbar_Imshow_Colorbar_Plot_Preset
from pykato.function import polka, vortex, checkers, sinusoid, gauss2d, text

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

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
        polka_test_image = polka((100, 100), 4, (20, 20), (-20, -20))
        checkers_test_image = checkers((100, 100), (25, 25), (-12.5, 12.5))
        sinusoid_test_image = sinusoid((100, 100), 20, 270, 45)
        gauss_test_image = gauss2d((100, 100), 0, 1, (20, 20), (50, 50))
        vortex_test_image = np.pi * (vortex((100, 100), 2) * 2 - 1)
        text01_test_image = text((100, 100), "01")
        text02_test_image = text((100, 100), "02")
        text03_test_image = text((100, 100), "03")
        text04_test_image = text((100, 100), "04")
        polka_vortex_complex_test_image = polka_test_image * np.exp(1j * vortex_test_image)
        checkers_vortex_complex_test_image = checkers_test_image * np.exp(1j * vortex_test_image)
        sinusoid_vortex_complex_test_image = sinusoid_test_image * np.exp(1j * vortex_test_image)
        gauss_vortex_complex_test_image = gauss_test_image * np.exp(1j * vortex_test_image)
        text01_vortex_complex_test_image = text01_test_image * np.exp(1j * vortex_test_image)
        text02_vortex_complex_test_image = text02_test_image * np.exp(1j * vortex_test_image)
        text03_vortex_complex_test_image = text03_test_image * np.exp(1j * vortex_test_image)
        text04_vortex_complex_test_image = text04_test_image * np.exp(1j * vortex_test_image)
        ω_data = np.linspace(0, 2 * np.pi, 100)
        self.data = {"polka_test_image": polka_test_image, "checkers_test_image": checkers_test_image, "sinusoid_test_image": sinusoid_test_image, "gauss_test_image": gauss_test_image, "vortex_test_image": vortex_test_image, "polka_vortex_complex_test_image": polka_vortex_complex_test_image, "checkers_vortex_complex_test_image": checkers_vortex_complex_test_image, "sinusoid_vortex_complex_test_image": sinusoid_vortex_complex_test_image, "gauss_vortex_complex_test_image": gauss_vortex_complex_test_image, "ω_data": ω_data, "text01_test_image": text01_test_image, "text02_test_image": text02_test_image, "text03_test_image": text03_test_image, "text04_test_image": text04_test_image, "text01_vortex_complex_test_image": text01_vortex_complex_test_image, "text02_vortex_complex_test_image": text02_vortex_complex_test_image, "text03_vortex_complex_test_image": text03_vortex_complex_test_image, "text04_vortex_complex_test_image": text04_vortex_complex_test_image}


class TestPreset(BaseTestPreset):
    def test_Imshow_Preset(self):
        imshow_preset = Imshow_Preset(self.data["polka_test_image"])
        imshow_preset.get_image().set_clim(0, 1)

        self.assertIsInstance(imshow_preset, Figure, "Figure not created by preset.Imshow_Preset")
        self.assertIsInstance(imshow_preset.get_imshow_ax(), Axes, "Imshow Axes not available in Figure created by preset.Imshow_Preset")
        self.assertIsInstance(imshow_preset.get_image(), AxesImage, "Imshow AxesImage not available in Figure created by preset.Imshow_Preset")

        imshow_preset.savefig("tests/output/test_Imshow_Preset.png")

    def test_Complex_Imshow_Preset(self):
        complex_imshow_preset = Complex_Imshow_Preset(self.data["polka_vortex_complex_test_image"])

        self.assertIsInstance(complex_imshow_preset, Figure, "Figure not created by preset.Complex_Imshow_Preset")
        self.assertIsInstance(complex_imshow_preset.get_imshow_ax(), Axes, "Imshow axes not available in Figure created by preset.Complex_Imshow_Preset")
        self.assertIsInstance(complex_imshow_preset.get_image(), AxesImage, "Imshow image not available in Figure created by preset.Complex_Imshow_Preset")

        complex_imshow_preset.savefig("tests/output/test_Imshow_Preset.png")

        complex_imshow_preset.get_image().set_data(self.data["checkers_vortex_complex_test_image"])
        complex_imshow_preset.savefig("tests/output/test_Imshow_Preset_set_data.png")
        complex_imshow_preset.get_imshow_ax().set_xlim((25, 75))
        complex_imshow_preset.get_imshow_ax().set_ylim((25, 75))
        complex_imshow_preset.savefig("tests/output/test_Imshow_Preset_set_xlim_set_ylim.png")

    def test_Imshow_Colorbar_Preset(self):
        imshow_colorbar_preset = Imshow_Colorbar_Preset(self.data["polka_test_image"])
        imshow_colorbar_preset.get_image().set_clim(0, 1)

        self.assertIsInstance(imshow_colorbar_preset, Figure, "Figure not created by preset.Imshow_Colorbar_Preset")

        imshow_ax, colorbar_ax = imshow_colorbar_preset.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow Axes not available in Figure created by preset.Imshow_Colorbar_Preset")
        self.assertIsInstance(colorbar_ax, Axes, "Colorbar Axes not available in Figure created by preset.Imshow_Colorbar_Preset")

        self.assertIsInstance(imshow_colorbar_preset.get_image(), AxesImage, "Image AxesImage not available in Figure created by preset.Imshow_Colorbar_Preset")
        self.assertIsInstance(imshow_colorbar_preset.get_imshow_ax(), Axes, "Imshow Axes not available in Figure created by preset.Imshow_Colorbar_Preset")
        self.assertIsInstance(imshow_colorbar_preset.get_cbar_ax(), Axes, "Colorbar Axes not available in Figure created by preset.Imshow_Colorbar_Preset")
        imshow_colorbar_preset.savefig("tests/output/test_Imshow_Colorbar_Preset.png")

        imshow_colorbar_preset.get_imshow_ax().set_xlim((25, 75))
        imshow_colorbar_preset.get_imshow_ax().set_ylim((25, 75))
        imshow_colorbar_preset.savefig("tests/output/test_Imshow_Colorbar_Preset_set_xlim_set_ylim.png")

    def test_Complex_Imshow_TwoColorbars_Preset(self):
        complex_imshow_two_colorbars_preset = Complex_Imshow_TwoColorbars_Preset(self.data["polka_vortex_complex_test_image"])

        self.assertIsInstance(complex_imshow_two_colorbars_preset, Figure, "Figure not created by preset.Complex_Imshow_TwoColorbars_Preset")

        imshow_ax, mod_colorbar_ax, arg_colorbar_ax = complex_imshow_two_colorbars_preset.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow Axes not available in Figure created by preset.Complex_Imshow_TwoColorbars_Preset")
        self.assertIsInstance(mod_colorbar_ax, Axes, "Colorbar Axes not available in Figure created by preset.Complex_Imshow_TwoColorbars_Preset")
        self.assertIsInstance(arg_colorbar_ax, Axes, "Colorbar Axes not available in Figure created by preset.Complex_Imshow_TwoColorbars_Preset")
        self.assertIsInstance(complex_imshow_two_colorbars_preset.get_imshow_ax(), Axes, "Imshow Axes not available in Figure created by preset.Complex_Imshow_TwoColorbars_Preset")

        complex_imshow_two_colorbars_preset.get_image().set_data(self.data["checkers_vortex_complex_test_image"])
        complex_imshow_two_colorbars_preset.savefig("tests/output/test_Complex_Imshow_TwoColorbars_Preset_set_data.png")
        complex_imshow_two_colorbars_preset.get_imshow_ax().set_xlim((25, 75))
        complex_imshow_two_colorbars_preset.get_imshow_ax().set_ylim((25, 75))
        complex_imshow_two_colorbars_preset.savefig("tests/output/test_Complex_Imshow_TwoColorbars_Preset_set_xlim_set_ylim.png")

    def test_ImageGrid_Preset(self):
        image_grid_preset = ImageGrid_Preset([self.data[key] for key in ("polka_test_image", "checkers_test_image", "gauss_test_image", "sinusoid_test_image")])

        self.assertIsInstance(image_grid_preset, Figure, "Figure not created by preset.ImageGrid_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax = image_grid_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Preset")

        images = image_grid_preset.get_images()
        for img in images:
            self.assertIsInstance(img, AxesImage, "Imshow List[AxesImage] not available in Figure created by preset.ImageGrid_Preset")

        image_grid_preset.savefig("tests/output/test_ImageGrid_Preset.png")

        image_grid_preset.get_images()[0].set_data(self.data["text01_test_image"])
        image_grid_preset.get_images()[1].set_data(self.data["text02_test_image"])
        image_grid_preset.get_images()[2].set_data(self.data["text03_test_image"])
        image_grid_preset.get_images()[3].set_data(self.data["text04_test_image"])

        image_grid_preset.savefig("tests/output/test_ImageGrid_Preset_set_data.png")

        x0 = 25 * np.sin(self.data["ω_data"]) + 50
        y1 = 25 * np.cos(2 * self.data["ω_data"] + np.pi / 2) + 50
        y2 = 25 * np.cos(3 * self.data["ω_data"]) + 50
        y3 = 25 * np.cos(4 * self.data["ω_data"] + np.pi / 2) + 50
        y4 = 25 * np.cos(5 * self.data["ω_data"]) + 50

        image_grid_preset.plot([[x0, y1], [x0, y2], [x0, y3], [x0, y4]])

        image_grid_preset.savefig("tests/output/test_ImageGrid_Preset_plot.png")

    def test_ImageGrid_Colorbar_Preset(self):
        image_grid_colorbar_preset = ImageGrid_Colorbar_Preset(tuple(self.data[key] for key in ("polka_test_image", "checkers_test_image", "gauss_test_image", "sinusoid_test_image")))

        self.assertIsInstance(image_grid_colorbar_preset, Figure, "Figure not created by preset.ImageGrid_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, colorbar_ax = image_grid_colorbar_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Colorbar_Preset")
        self.assertIsInstance(colorbar_ax, Axes, "Imshow Axes not available in Figure created by preset.ImageGrid_Colorbar_Preset")

        images = image_grid_colorbar_preset.get_images()
        for img in images:
            self.assertIsInstance(img, AxesImage, "Imshow List[AxesImage] not available in Figure created by preset.ImageGrid_Colorbar_Preset")

        image_grid_colorbar_preset.savefig("tests/output/test_ImageGrid_Colorbar_Preset.png")

        image_grid_colorbar_preset.get_images()[0].set_data(self.data["text01_test_image"])
        image_grid_colorbar_preset.get_images()[1].set_data(self.data["text02_test_image"])
        image_grid_colorbar_preset.get_images()[2].set_data(self.data["text03_test_image"])
        image_grid_colorbar_preset.get_images()[3].set_data(self.data["text04_test_image"])

        image_grid_colorbar_preset.savefig("tests/output/test_ImageGrid_Colorbar_Preset_set_data.png")

        x0 = 25 * np.sin(self.data["ω_data"]) + 50
        y1 = 25 * np.cos(2 * self.data["ω_data"] + np.pi / 2) + 50
        y2 = 25 * np.cos(3 * self.data["ω_data"]) + 50
        y3 = 25 * np.cos(4 * self.data["ω_data"] + np.pi / 2) + 50
        y4 = 25 * np.cos(5 * self.data["ω_data"]) + 50

        image_grid_colorbar_preset.plot([[x0, y1], [x0, y2], [x0, y3], [x0, y4]])

        image_grid_colorbar_preset.savefig("tests/output/test_ImageGrid_Colorbar_Preset_plot.png")

    def test_Complex_ImageGrid_TwoColorbars_Preset(self):
        complex_image_grid_two_colorbars_preset = Complex_ImageGrid_TwoColorbars_Preset(tuple(self.data[key] for key in ("polka_vortex_complex_test_image", "checkers_vortex_complex_test_image", "sinusoid_vortex_complex_test_image", "gauss_vortex_complex_test_image")))

        self.assertIsInstance(complex_image_grid_two_colorbars_preset, Figure, "Figure not created by preset.Complex_ImageGrid_TwoColorbars_Preset")

        imshow1_ax, imshow2_ax, imshow3_ax, imshow4_ax, colorbar1_ax, colorbar2_ax = complex_image_grid_two_colorbars_preset.get_axes()
        self.assertIsInstance(imshow1_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow2_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow3_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(imshow4_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(colorbar1_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")
        self.assertIsInstance(colorbar2_ax, Axes, "Imshow axis not available in Figure created by preset.Complex_ImageGrid_TwoColorbars_Preset")

        images = complex_image_grid_two_colorbars_preset.get_images()
        for img in images:
            self.assertIsInstance(img, AxesImage, "Imshow List[AxesImage] not available in Figure created by preset.ImageGrid_Colorbar_Preset")

        complex_image_grid_two_colorbars_preset.savefig("tests/output/test_Complex_ImageGrid_TwoColorbars_Preset.png")

        complex_image_grid_two_colorbars_preset.get_images()[0].set_data(self.data["text01_vortex_complex_test_image"])
        complex_image_grid_two_colorbars_preset.get_images()[1].set_data(self.data["text02_vortex_complex_test_image"])
        complex_image_grid_two_colorbars_preset.get_images()[2].set_data(self.data["text03_vortex_complex_test_image"])
        complex_image_grid_two_colorbars_preset.get_images()[3].set_data(self.data["text04_vortex_complex_test_image"])

        complex_image_grid_two_colorbars_preset.savefig("tests/output/test_Complex_ImageGrid_TwoColorbars_Preset_set_data.png")

        x0 = 25 * np.sin(self.data["ω_data"]) + 50
        y1 = 25 * np.cos(2 * self.data["ω_data"] + np.pi / 2) + 50
        y2 = 25 * np.cos(3 * self.data["ω_data"]) + 50
        y3 = 25 * np.cos(4 * self.data["ω_data"] + np.pi / 2) + 50
        y4 = 25 * np.cos(5 * self.data["ω_data"]) + 50

        complex_image_grid_two_colorbars_preset.plot([[x0, y1], [x0, y2], [x0, y3], [x0, y4]])

        complex_image_grid_two_colorbars_preset.savefig("tests/output/test_Complex_ImageGrid_TwoColorbars_Preset_plot.png")

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
