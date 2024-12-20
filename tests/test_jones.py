import unittest
import numpy as np
import matplotlib.pyplot as plt
from pykato.plotfunction.jones import Phaser_Preset
from pykato.jones import hlp, vlp, lcp, rcp, lp, linear_polarizer
from matplotlib.axis import Axis


def style_ax(ax: Axis):
    ax.set_aspect(1)
    ax.set_xlim(-1.2, 1.2)
    ax.set_xticks(np.linspace(-1, 1, 3))
    ax.set_xticklabels([])
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks(np.linspace(-1, 1, 3))
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.grid(visible=None, which="major")


class TestJones(unittest.TestCase):
    # pylint: disable=missing-class-docstring

    def test_horizontal_linear_polarizer(self):
        polarizer = linear_polarizer(0)
        for actual_pair, expected_pair in zip(polarizer, ((1, 0), (0, 0))):
            for actual_value, expected_value in zip(actual_pair, expected_pair):
                self.assertAlmostEqual(actual_value, expected_value, places=9)

    def test_vertical_linear_polarizer(self):
        polarizer = linear_polarizer(np.pi / 2)
        for actual_pair, expected_pair in zip(polarizer, ((0, 0), (0, 1))):
            for actual_value, expected_value in zip(actual_pair, expected_pair):
                self.assertAlmostEqual(actual_value, expected_value, places=9)

    def test_Phaser_preset(self):

        fig, ((ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13)) = plt.subplots(2, 4)

        style_ax(ax00)
        Phaser_Preset(ax00, (hlp()), stops=[0.25])

        style_ax(ax01)
        Phaser_Preset(ax01, (vlp()), stops=[0.25])

        style_ax(ax02)
        Phaser_Preset(ax02, (rcp()))

        style_ax(ax03)
        Phaser_Preset(ax03, (lcp()))

        style_ax(ax10)
        Phaser_Preset(ax10, (lp(np.pi / 4)), stops=[0.25])

        style_ax(ax11)
        Phaser_Preset(ax11, np.dot(vlp(), linear_polarizer(0)), stops=[0.25])

        fig.savefig("tests/output/preset_test_Phaser_Preset.png")
