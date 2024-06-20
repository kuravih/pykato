import unittest
from pykato.plotfunction.gridspec_layout import GridSpec_Layout
from pykato.plotfunction import preset
from pykato.function import checkers, sinusoid, vortex, box, circle, gauss2d, airy, polka, linear2d, least_squares_fit_2d, gauss2d_fn, linear2d_fn, least_squares_fit, generate_coordinates
import numpy as np


class TestFunction(unittest.TestCase):
    # pylint: disable=missing-class-docstring

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

    def test_gauss2d(self):
        data = gauss2d((200, 200), offset=0, height=1, width=(10, 10), center=(100, 100), tilt=0)
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.Gauss2d()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_gauss2d.png")

    def test_airy(self):
        data = airy((200, 200), center=(100, 100), radius=0.25, height=100)
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.Airy()")

        figure = preset.Imshow_Preset(np.log10(data / np.max(data)))
        figure.savefig("tests/output/function_airy.png")

    def test_polka(self):
        data = polka((200, 200), 3, (40, 40), (0, 0))
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.polka()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_polka.png")

    def test_linear2D(self):
        data = linear2d((200, 200), 3, 2, 1)
        self.assertIsInstance(data, np.ndarray, "Array not returned by function.Linear2d()")

        figure = preset.Imshow_Preset(data)
        figure.savefig("tests/output/function_linear2d.png")

    def test_least_squares_fit_2d_gauss2d(self):
        offset, height, width, center = 0, 1, (25, 25), (100, 100)
        gauss2d_data = gauss2d((200, 200), offset=offset, height=height, width=width, center=center, tilt=0)

        def fit_gauss_2d_fn(xx_yy, cx, cy, o, h, wx, wy):
            return gauss2d_fn(xx_yy, (cx, cy), o, h, (wx, wy))

        (cx, cy, fit_offset, fit_height, wx, wy), _ = least_squares_fit_2d(gauss2d_data, fit_gauss_2d_fn, guess_prms=(101, 101, 0, 1.1, 25.5, 25.5)) # pylint: disable=unbalanced-tuple-unpacking
        fit_center = (cx, cy)
        fit_width = (wx, wy)

        self.assertAlmostEqual(offset, fit_offset, delta=0.0001, msg="offset estimation not accurate")
        self.assertAlmostEqual(height, fit_height, delta=0.0001, msg="height estimation not accurate")
        for val, fit_val in zip(center, fit_center):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="center estimation not accurate")
        for val, fit_val in zip(width, fit_width):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="width estimation not accurate")

        fit_gauss2d_data = gauss2d((200, 200), offset=fit_offset, height=fit_height, width=fit_width, center=fit_center, tilt=0)

        delta = gauss2d_data - fit_gauss2d_data
        log_delta = np.log10(np.abs(delta))
        figure = preset.Imshow_Colorbar_Preset(log_delta)
        figure.get_image().set_clim(-16, 0)
        figure.get_cbar_ax().set_ylabel("Error")
        figure.savefig("tests/output/delta_least_squares_fit_2d_gauss2d.png")

    def test_least_squares_fit_2d_linear2d(self):
        a, b, c = 3, 2, 1
        x_coord, y_coord = generate_coordinates((200, 200), (-10, -10), cartesian=True)
        linear2d_data = linear2d_fn((x_coord, y_coord), a, b, c)

        (fit_a, fit_b, fit_c), _ = least_squares_fit_2d(linear2d_data, linear2d_fn, xy_coords=(x_coord, y_coord), guess_prms=(3.3, 2.2, 1.1)) # pylint: disable=unbalanced-tuple-unpacking

        for val, fit_val in zip((a, b, c), (fit_a, fit_b, fit_c)):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="parameter estimation not accurate")

        fit_linear2d_data = linear2d_fn((x_coord, y_coord), fit_a, fit_b, fit_c)

        delta = linear2d_data - fit_linear2d_data
        log_delta = np.log10(np.abs(delta))
        figure = preset.Imshow_Colorbar_Preset(log_delta)
        figure.get_image().set_clim(-16, 0)
        figure.get_cbar_ax().set_ylabel("Error")
        figure.savefig("tests/output/delta_least_squares_fit_2d_linear2d.png")

    def test_least_squares_fit_linear(self):
        def fit_linear_fn(x, m, c):
            return m * x + c

        x = np.linspace(0, 10, 21)
        m, c = 0.25, -1
        data = fit_linear_fn(x, m, c)

        (fit_m, fit_c), _ = least_squares_fit(data, fit_linear_fn, guess_prms=(0.26, -1.1), x_coord=x) # pylint: disable=unbalanced-tuple-unpacking

        for val, fit_val in zip((m, c), (fit_m, fit_c)):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="parameter estimation not accurate")

        fit_data = fit_linear_fn(x, fit_m, fit_c)

        for val, fit_val in zip(data, fit_data):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="fit estimation not accurate")

        error = np.abs(data - fit_data)
        log_error = np.log10(error)
        figure = GridSpec_Layout(1, 1)
        (imshow_ax,) = figure.get_axes()
        imshow_ax.plot(x, log_error, "+", markersize=10)
        imshow_ax.set_ylim(-16, -15)
        imshow_ax.set_ylabel("error")
        imshow_ax.set_xlim(0, 10)
        imshow_ax.set_xlabel("x")
        figure.savefig("tests/output/test_least_squares_fit_linear.png")


    def test_least_squares_fit_quadratic(self):
        def fit_quadratic_fn(x, a, b, c):
            return a * x * x + b * x + c

        x = np.linspace(0, 10, 21)
        a, b, c = 0.25, -1, 2.5
        data = fit_quadratic_fn(x, a, b, c)

        (fit_a, fit_b, fit_c), _ = least_squares_fit(data, fit_quadratic_fn, guess_prms=(0.26, -1.1, 2.6), x_coord=x) # pylint: disable=unbalanced-tuple-unpacking

        for val, fit_val in zip((a, b, c), (fit_a, fit_b, fit_c)):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="parameter estimation not accurate")

        fit_data = fit_quadratic_fn(x, fit_a, fit_b, fit_c)

        for val, fit_val in zip(data, fit_data):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="fit estimation not accurate")

        error = np.abs(data - fit_data)
        log_error = np.log10(error)
        figure = GridSpec_Layout(1, 1)
        (imshow_ax,) = figure.get_axes()
        imshow_ax.plot(x, log_error, "+", markersize=10)
        imshow_ax.set_ylim(-16, -15)
        imshow_ax.set_ylabel("error")
        imshow_ax.set_xlim(0, 10)
        imshow_ax.set_xlabel("x")
        figure.savefig("tests/output/test_least_squares_fit_quadratic.png")

    def test_least_squares_fit_sinusoid(self):
        def fit_sinusoid_fn(x, a, b, c, d):
            return a * np.sin(b * x + c) + d

        x = np.linspace(0, 10, 21)
        a, b, c, d = 0.25, -1, 2.5, 1
        data = fit_sinusoid_fn(x, a, b, c, d)

        (fit_a, fit_b, fit_c, fit_d), _ = least_squares_fit(data, fit_sinusoid_fn, guess_prms=(0.26, -1.1, 2.6, 1.1), x_coord=x) # pylint: disable=unbalanced-tuple-unpacking

        for val, fit_val in zip((a, b, c, d), (fit_a, fit_b, fit_c, fit_d)):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="parameter estimation not accurate")

        fit_data = fit_sinusoid_fn(x, fit_a, fit_b, fit_c, fit_d)

        for val, fit_val in zip(data, fit_data):
            self.assertAlmostEqual(val, fit_val, delta=0.0001, msg="fit estimation not accurate")

        error = np.abs(data - fit_data)
        log_error = np.log10(error)
        figure = GridSpec_Layout(1, 1)
        (imshow_ax,) = figure.get_axes()
        imshow_ax.plot(x, log_error, "+", markersize=10)
        imshow_ax.set_ylim(-16, -15)
        imshow_ax.set_ylabel("error")
        imshow_ax.set_xlim(0, 10)
        imshow_ax.set_xlabel("x")
        figure.savefig("tests/output/least_squares_fit_sinusoid.png")
