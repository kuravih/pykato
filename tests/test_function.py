import unittest
from pykato.plotfunction import preset
from pykato.function import checkers, sinusoid, vortex, box, circle, Gauss2d, Airy, polka
import numpy as np


class TestFunction(unittest.TestCase):
    def test_checkers(self):
        data = checkers((100, 100), (25, 25), (-12.5, 12.5))
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.checkers()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_checkers.png")

    def test_sinusoid(self):
        data = sinusoid((200, 200), 25, 90, 45)
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.sinusoid()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_sinusoid.png")

    def test_vortex(self):
        arg = (2 * vortex((200, 200), 3) - 1) * np.pi
        self.assertIsInstance(arg, np.ndarray, "Array not returned by function.vortex()")

        polka_test_image = polka((100, 100), 4, (20, 20), (-10, -10))
        vortex_test_image = np.pi * (vortex((100, 100), 2) * 2 - 1)
        polka_vortex_complex_test_image = polka_test_image * np.exp(1j * vortex_test_image)
        figure = preset.Complex_Imshow_Preset(polka_vortex_complex_test_image)
        figure.savefig("tests/output/function_vortex.png")

    def test_box(self):
        data = box((200, 200), (40, 50), (-100, -100))
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.box()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_box.png")

    def test_circle(self):
        data = circle((200, 200), 50, (-100, -100))
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.circle()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_circle.png")

    def test_Gauss2d(self):
        data = Gauss2d((200, 200), offset=0, height=1, width=(3, 3), center=(100, 100), tilt=0)
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.Gauss2d()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_Gauss2d.png")

    def test_Airy(self):
        data = Airy((200, 200), center=(100, 100), radius=100, height=10)
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.Airy()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_Airy.png")

    def test_polka(self):
        data = polka((200, 200), 3, (40, 40), (0, 0))
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.polka()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_polka.png")
