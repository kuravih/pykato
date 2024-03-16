import unittest
from pykato.plotfunction import gridspec_layout, diagram
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


class TestGridSpecLayout(unittest.TestCase):
    def test_Simple_Layout(self):
        figure = gridspec_layout.Simple_Layout()
        self.assertIsInstance(figure, Figure, "Figure not created by gridspec_layout.Simple_Layout")

        (imshow_ax,) = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.Simple_Layout")

        figure.savefig("tests/output/gridspec_layout_Simple_Layout.png")

    def test_Image_Layout(self):
        figure = gridspec_layout.Image_Layout()
        self.assertIsInstance(figure, Figure, "Figure not created by gridspec_layout.Image_Layout")

        (imshow_ax,) = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.Image_Layout")

        figure.savefig("tests/output/gridspec_layout_Image_Layout.png")

    def test_Image_Colorbar_Layout(self):
        figure = gridspec_layout.Image_Colorbar_Layout()
        self.assertIsInstance(figure, Figure, "Figure not created by gridspec_layout.Image_Colorbar_Layout")

        imshow_ax, colorbar_ax = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.Image_Colorbar_Layout")
        self.assertIsInstance(colorbar_ax, Axes, "colorbar_ax Axis not available in Figure created by gridspec_layout.Image_Colorbar_Layout")

        figure.savefig("tests/output/gridspec_layout_Image_Colorbar_Layout.png")

    def test_Image_Colorbar_Colorbar_Layout(self):
        figure = gridspec_layout.Image_Colorbar_Colorbar_Layout()
        self.assertIsInstance(figure, Figure, "Figure not created by gridspec_layout.Image_Colorbar_Colorbar_Layout")

        imshow_ax, colorbar1_ax, colorbar2_ax = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "imshow_ax Axis not available in Figure created by gridspec_layout.Image_Colorbar_Colorbar_Layout")
        self.assertIsInstance(colorbar1_ax, Axes, "colorbar_ax Axis not available in Figure created by gridspec_layout.Image_Colorbar_Colorbar_Layout")
        self.assertIsInstance(colorbar2_ax, Axes, "colorbar_ax Axis not available in Figure created by gridspec_layout.Image_Colorbar_Colorbar_Layout")

        figure.savefig("tests/output/gridspec_layout_Image_Colorbar_Colorbar_Layout.png")


class TestDiagram(unittest.TestCase):
    def test_Image_Diagram_exclude_figure(self):
        data = np.zeros((200, 200))

        figure = diagram.Image_Diagram(data)
        self.assertIsInstance(figure, Figure, "Figure not created by diagram.Image_Diagram")

        (imshow_ax,) = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Axis not available in Figure created by diagram.Image_Diagram")

        figure.savefig("tests/output/diagram_Image_Diagram_exclude_figure.png")

    def test_Image_Diagram_include_figure(self):
        data = np.zeros((200, 200))

        figure = diagram.Image_Layout()
        figure = diagram.Image_Diagram(data, figure)
        self.assertIsInstance(figure, Figure, "Figure not created by diagram.Image_Diagram")

        (imshow_ax,) = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by diagram.Image_Diagram")

        figure.savefig("tests/output/diagram_Image_Diagram_include_figure.png")

    def test_Image_Colorbar_Diagram_exclude_figure(self):
        data = np.zeros((200, 200))

        figure = diagram.Image_Colorbar_Diagram(data)
        figure.resize(None)
        self.assertIsInstance(figure, Figure, "Figure not created by diagram.Image_Colorbar_Diagram")

        imshow_ax, colorbar_ax = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by diagram.Image_Diagram")
        self.assertIsInstance(colorbar_ax, Axes, "Colorbar axis not available in Figure created by diagram.Image_Diagram")

        figure.savefig("tests/output/diagram_Image_Colorbar_Diagram_exclude_figure.png")

    def test_Image_Colorbar_Diagram_include_figure(self):
        data = np.zeros((200, 200))

        figure = diagram.Image_Colorbar_Layout()
        figure = diagram.Image_Colorbar_Diagram(data)
        figure.resize(None)
        self.assertIsInstance(figure, Figure, "Figure not created by diagram.Image_Colorbar_Diagram")

        imshow_ax, colorbar_ax = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by diagram.Image_Diagram")
        self.assertIsInstance(colorbar_ax, Axes, "Colorbar axis not available in Figure created by diagram.Image_Diagram")

        figure.savefig("tests/output/diagram_Image_Colorbar_Diagram_include_figure.png")

    def test_Complex_Diagram_exlcude_figure(self):
        data = np.zeros((200, 200), dtype=np.complex64)

        figure = diagram.Complex_Diagram(data)
        figure.resize(None)
        self.assertIsInstance(figure, Figure, "Figure not created by diagram.Complex_Diagram")

        imshow_ax, colorbar_mod_ax, colorbar_arg_ax = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by diagram.Complex_Diagram")
        self.assertIsInstance(colorbar_mod_ax, Axes, "Colorbar_mod_ax axis not available in Figure created by diagram.Complex_Diagram")
        self.assertIsInstance(colorbar_arg_ax, Axes, "Colorbar_arg_ax axis not available in Figure created by diagram.Complex_Diagram")

        figure.savefig("tests/output/diagram_Complex_Diagram_exclude_figure.png")

    def test_Complex_Diagram_include_figure(self):
        data = np.zeros((200, 200), dtype=np.complex64)

        figure = diagram.Image_Colorbar_Colorbar_Layout()
        figure = diagram.Complex_Diagram(data)
        figure.resize(None)
        self.assertIsInstance(figure, Figure, "Figure not created by diagram.Complex_Diagram")

        imshow_ax, colorbar_mod_ax, colorbar_arg_ax = figure.get_axes()
        self.assertIsInstance(imshow_ax, Axes, "Imshow axis not available in Figure created by diagram.Complex_Diagram")
        self.assertIsInstance(colorbar_mod_ax, Axes, "Colorbar_mod_ax axis not available in Figure created by diagram.Complex_Diagram")
        self.assertIsInstance(colorbar_arg_ax, Axes, "Colorbar_arg_ax axis not available in Figure created by diagram.Complex_Diagram")

        figure.savefig("tests/output/diagram_Complex_Diagram_include_figure.png")
