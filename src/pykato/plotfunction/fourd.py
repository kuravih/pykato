from typing import Optional
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from pykato.plotfunction import preset
from pykato.fourd import Measurement
import numpy as np


def Measurement_Preset(data: Measurement, mask: Optional[np.ndarray] = None, figure: Optional[Figure] = None) -> Figure:
    """
    FourD measurement layout

    Examples:
        figure = Measurement_Preset(data)

    Parameters:
        data: Measurement
            FourD measurement.
        mask: Optional[np.ndarray] = None
            Mask
        figure: Optional[Figure] = None
            Figure object.

    Returns: Figure
        Figure object with the imshow axis.

    Functions:
        close():
            Properly close the figure.
    """

    if mask is None:
        mask = data.surface.mask

    data_min = -200  # np.min(ref_flat.surface)
    data_max = 200  # np.max(ref_flat.surface)

    figure = preset.Imshow_Colorbar_Preset(data.surface, figure)

    figure.get_imshow_ax().imshow(mask * 0.0, alpha=mask * 0.5, cmap="gray")
    figure.get_imshow_ax().invert_yaxis()
    figure.get_image().set_clim(data_min, data_max)
    figure.get_image().set_cmap("bwr")
    figure.get_cbar_ax().set_title("nm", size=8)
    figure.get_cbar_ax().tick_params(axis="y", labelsize=8)
    figure.get_imshow_ax().set_xlabel("px", size=8)
    figure.get_imshow_ax().set_ylabel("px", size=8)
    figure.get_imshow_ax().text(1950, 1950, f"ptv : {np.ptp(data.surface*mask):6.2f} nm\nrms : {np.std(data.surface*mask):6.2f} nm", fontfamily="monospace", horizontalalignment="right", verticalalignment="top", size=8, bbox=dict(boxstyle="round", facecolor="white", pad=0.25, linewidth=0.75))
    figure.get_imshow_ax().tick_params(axis="y", labelsize=8)
    figure.get_imshow_ax().tick_params(axis="x", labelsize=8)

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Measurement_Zernike_Preset(data: Measurement, mask: Optional[np.ndarray] = None, figure: Optional[Figure] = None) -> Figure:
    """
    FourD measurement layout with zernike plot

    Examples:
        figure = Measurement_Zernike_Preset(data)

    Parameters:
        data: Measurement
            FourD measurement.
        mask: Optional[np.ndarray] = None
            Mask
        figure: Optional[Figure] = None
            Figure object.

    Returns: plt.Figure
        Figure object with the imshow axis.

    Functions:
        get_image(): AxesImage
            Get imshow image.

        get_bars(): Axis
            Get imshow axis.

        close():
            Properly close the figure.
    """

    if mask is None:
        mask = data.surface.mask

    data_min = -200  # np.min(ref_flat.surface)
    data_max = 200  # np.max(ref_flat.surface)

    figure = preset.Imshow_Colorbar_Imshow_Colorbar_Plot_Preset((data.surface, data.surface * np.nan), figure)

    figure.get_imshow_axes()[0].imshow(mask * 0.0, alpha=mask * 0.5, cmap="gray")
    figure.get_imshow_axes()[0].invert_yaxis()
    figure.get_images()[0].set_clim(data_min, data_max)
    figure.get_images()[0].set_cmap("bwr")
    figure.get_cbar_axes()[0].set_title("nm", size=8)
    figure.get_cbar_axes()[0].tick_params(axis="y", labelsize=8)
    figure.get_imshow_axes()[0].set_xlabel("px", size=8)
    figure.get_imshow_axes()[0].set_ylabel("px", size=8)
    figure.get_imshow_axes()[0].text(1950, 1950, f"ptv : {np.ptp(data.surface*~mask):6.2f} nm\nrms : {np.std(data.surface*~mask):6.2f} nm", fontfamily="monospace", horizontalalignment="right", verticalalignment="top", size=8, bbox=dict(boxstyle="round", facecolor="white", pad=0.25, linewidth=0.75))
    figure.get_imshow_axes()[0].tick_params(axis="y", labelsize=8)
    figure.get_imshow_axes()[0].tick_params(axis="x", labelsize=8)

    figure.get_imshow_axes()[1].set_axis_off()
    figure.get_cbar_axes()[1].set_visible(False)
    figure.get_imshow_axes()[1].text(500, 1950, f"Hexapod Position\n" + f"X : {np.pi:6.2f} nm\n" + f"Y : {np.pi:6.2f} nm\n" + f"Z : {np.pi:6.2f} nm\n" + f"U : {np.pi:6.2f} nm\n" + f"V : {np.pi:6.2f} nm\n" + f"W : {np.pi:6.2f} nm", fontfamily="monospace", horizontalalignment="left", verticalalignment="top", size=10, bbox=dict(boxstyle="round", facecolor="white", pad=0.25, linewidth=0.75))

    zernike_labels = ("Piston", "TiltX", "TiltY", "Power", "AstigX", "Astig45", "ComaX", "ComaY", "Spherical")

    figure.get_plot_ax().axhline(alpha=0.25, linewidth=0.5, color="black")
    bars = figure.get_plot_ax().bar(zernike_labels, data.calculated_zernike_terms)
    figure.get_plot_ax().tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.get_plot_ax().tick_params(axis="y", labelsize=8)
    figure.get_plot_ax().set_ylabel("nm", size=8)

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return figure.get_images()[0]

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_bars(): # TODO: hint
        return bars

    figure.get_bars = _get_bars
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure
