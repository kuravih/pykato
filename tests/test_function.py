import unittest
from pykato.plotfunction import preset
from pykato.function import checkers, sinusoid, vortex, box, circle, Gauss2d, Airy, polka, Linear2d, LeastSquareFit2d, Gauss2d_fn, Linear2d_fn, LeastSquareFit, generate_coordinates
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

    def test_Linear2D(self):
        data = Linear2d((200, 200), 3, 2, 1)
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.Linear2d()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_Linear2d.png")

    def test_LeastSquareFit2d_Gauss2d(self):
        offset, height, width, center = 0, 1, (25, 25), (100, 100)
        gauss2d_data = Gauss2d((200, 200), offset=offset, height=height, width=width, center=center, tilt=0)

        def fit_Gauss2d_fn(xxyy, cx, cy, o, h, wx, wy):
            return Gauss2d_fn(xxyy, (cx, cy), o, h, (wx, wy))

        (cx, cy, fit_offset, fit_height, wx, wy), _ = LeastSquareFit2d(gauss2d_data, fit_Gauss2d_fn, guess_prms=(101, 101, 0, 1.1, 25.5, 25.5))
        fit_center = (cx, cy)
        fit_width = (wx, wy)

        self.assertAlmostEqual(offset, fit_offset, delta=0.0001, msg="offset estimation not accurate")
        self.assertAlmostEqual(height, fit_height, delta=0.0001, msg="height estimation not accurate")
        for val, fit_val in zip(center, fit_center):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="center estimation not accurate")
        for val, fit_val in zip(width, fit_width):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="width estimation not accurate")

        fit_gauss2d_data = Gauss2d((200, 200), offset=fit_offset, height=fit_height, width=fit_width, center=fit_center, tilt=0)

        figure = preset.Imshow_Colorbar_Preset(gauss2d_data - fit_gauss2d_data)
        figure.savefig("tests/output/delta_LeastSquareFit2d_Gauss2d.png")

    def test_LeastSquareFit2d_Linear2d(self):
        a, b, c = 3, 2, 1
        x_coord, y_coord = generate_coordinates((200, 200), (-10, -10), cartesian=True)
        linear2d_data = Linear2d_fn((x_coord, y_coord), a, b, c)

        (fit_a, fit_b, fit_c), _ = LeastSquareFit2d(linear2d_data, Linear2d_fn, xy_coords=(x_coord, y_coord), guess_prms=(3.3, 2.2, 1.1))

        for val, fit_val in zip((a, b, c), (fit_a, fit_b, fit_c)):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="parameter estimation not accurate")

        fit_linear2d_data = Linear2d_fn((x_coord, y_coord), fit_a, fit_b, fit_c)

        figure = preset.Imshow_Colorbar_Preset(linear2d_data - fit_linear2d_data)
        figure.savefig("tests/output/delta_LeastSquareFit2d_Linear2d.png")

    def test_LeastSquareFit2d_Gauss2d(self):
        offset, height, width, center = 0, 1, (25, 25), (100, 100)
        gauss2d_data = Gauss2d((200, 200), offset=offset, height=height, width=width, center=center, tilt=0)

        def fit_Gauss2d_fn(xxyy, cx, cy, o, h, wx, wy):
            return Gauss2d_fn(xxyy, (cx, cy), o, h, (wx, wy))

        (cx, cy, fit_offset, fit_height, wx, wy), _ = LeastSquareFit2d(gauss2d_data, fit_Gauss2d_fn, guess_prms=(101, 101, 0, 1.1, 25.5, 25.5))
        fit_center = (cx, cy)
        fit_width = (wx, wy)

        self.assertAlmostEqual(offset, fit_offset, delta=0.0001, msg="offset estimation not accurate")
        self.assertAlmostEqual(height, fit_height, delta=0.0001, msg="height estimation not accurate")
        for val, fit_val in zip(center, fit_center):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="center estimation not accurate")
        for val, fit_val in zip(width, fit_width):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="width estimation not accurate")

        fit_gauss2d_data = Gauss2d((200, 200), offset=fit_offset, height=fit_height, width=fit_width, center=fit_center, tilt=0)

        figure = preset.Imshow_Colorbar_Preset(gauss2d_data - fit_gauss2d_data)
        figure.savefig("tests/output/delta_LeastSquareFit2d.png")

    def test_LeastSquareFit_Quadratic(self):
        def fit_quadratic_fn(x, a, b, c):
            return a * x * x + b * x + c

        x = np.linspace(0, 10, 21)
        a, b, c = 0.25, -1, 2.5
        data = fit_quadratic_fn(x, a, b, c)

        (fit_a, fit_b, fit_c), _ = LeastSquareFit(data, fit_quadratic_fn, guess_prms=(0.26, -1.1, 2.6), x_coord=x)

        for val, fit_val in zip((a, b, c), (fit_a, fit_b, fit_c)):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="width estimation not accurate")

        fit_data = fit_quadratic_fn(x, fit_a, fit_b, fit_c)

        for val, fit_val in zip(data, fit_data):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="width estimation not accurate")

    def test_LeastSquareFit_Sinusoid(self):
        def fit_sinusoid_fn(x, a, b, c, d):
            return a * np.sin(b * x + c) + d

        x = np.linspace(0, 10, 21)
        a, b, c, d = 0.25, -1, 2.5, 1
        data = fit_sinusoid_fn(x, a, b, c, d)

        (fit_a, fit_b, fit_c, fit_d), _ = LeastSquareFit(data, fit_sinusoid_fn, guess_prms=(0.26, -1.1, 2.6, 1.1), x_coord=x)

        for val, fit_val in zip((a, b, c, d), (fit_a, fit_b, fit_c, fit_d)):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="width estimation not accurate")

        fit_data = fit_sinusoid_fn(x, fit_a, fit_b, fit_c, fit_d)

        for val, fit_val in zip(data, fit_data):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="width estimation not accurate")
