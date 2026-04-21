import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from warnings import warn
import xarray as xa
import os
import pickle as pkl

from pytools import hhsignal, hhtools
from pytools import utils_fig as uf
import figure_manager as fm

uf.set_plt()

fdir_coburst = "../results/twopop_regime_samples/coburst"
fdir_simul = "../results/twopop_regime_samples/data"
fdir_simul_example = "../results/twopop_regime_samples"


def get_psd_set(detail):
    psd_set = []
    for i in range(2):
        psd, fpsd, tpsd = hhsignal.get_stfft(
            detail["vlfp"][i + 1], detail["ts"], 2000, frange=(5, 110)
        )
        psd_set.append(psd)
    return psd_set, fpsd, tpsd


def show_psd(
    ax, psd, fpsd, tpsd, vmin=0.0, vmax=1.0, cmap="jet", interpolation="bicubic"
):
    im_obj = ax.imshow(
        psd,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
        interpolation=interpolation,
    )

    return im_obj


def set_ticks(ax, tl, yl):
    plt.sca(ax)
    plt.xlim(tl)
    xt = np.arange(tl[0], tl[1] + 1e-3, 0.5)
    xt_labels = np.arange(1, 1 + tl[1] - tl[0] + 1e-3, 0.5)
    plt.xticks(xt, xt_labels)
    plt.yticks(np.arange(20, 101, 20))
    plt.ylim(yl)


def remove_small_components(binary, min_size):
    """
    Remove connected components smaller than a threshold.

    Parameters
    ----------
    binary : 2D array (0/1)
        Input binary image.
    min_size : int
        Components with size < min_size will be removed.

    Returns
    -------
    2D array
        Binary array where small components are set to 0.
    """
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    labeled, num_features = label(binary.astype(bool), structure=structure)

    if num_features == 0:
        return np.zeros_like(binary)

    counts = np.bincount(labeled.ravel())

    keep_mask = counts >= min_size
    keep_mask[0] = False

    result = keep_mask[labeled]
    return result.astype(binary.dtype)


@fm.figure_renderer("mfop_example", reset=False)
def draw_example(figsize=(9.5, 6), fdir_simul="", cid=7, nt=93, tl=(2.2, 4.2), th_q=95):

    with os.scandir(fdir_simul) as it:
        if any(it):
            summary_obj = hhtools.SummaryLoader(fdir_simul)
            detail = summary_obj.load_detail(cid - 1, nt)
        else:
            warn("Simulated data do not exist. Replace example dataset")
            f = os.path.join(fdir_simul_example, "detail_sample_%d%02d.pkl" % (cid, nt))
            if not os.path.exists(f):
                raise FileExistsError("Example data %s does not exist" % (f))
            with open(f, "rb") as fp:
                detail = pkl.load(fp)

    psd_set, fpsd, tpsd = get_psd_set(detail)
    psd_set = np.array(psd_set)

    idt = (tpsd >= tl[0] - 1) & (tpsd <= tl[1] + 1)
    psd_set_sub = psd_set[:, :, idt]
    tpsd_sub = tpsd[idt]

    th = np.percentile(psd_set, q=th_q, axis=2, keepdims=True)
    psd_th_sub = psd_set_sub > th
    for i in range(2):
        psd_th_sub[i] = remove_small_components(psd_th_sub[i], 3)

    yl = (18, 82)
    lb_set = (r"Pop$_{F}$", r"Pop$_{S}$")
    color_set = ("k", "#d25606")

    fig = uf.get_figure(figsize)
    axs = uf.get_custom_subplots(
        [1, 1], [1, 0.05, 1.0], w_blank_interval_set=[0.02, 0.2], h_blank_interval=0.2
    )

    for i in range(2):
        ax_psd = axs[i][0]
        im_obj = show_psd(
            ax_psd,
            psd_set_sub[i] - psd_set[i].mean(axis=1, keepdims=True),
            fpsd,
            tpsd_sub,
            vmin=-0.3,
            vmax=0.3,
        )
        ax_psd.set_ylabel("Frequency (Hz)")
        plt.colorbar(im_obj, cax=axs[i][1], ticks=(-0.3, 0, 0.3))
        set_ticks(ax_psd, tl, yl)

        ax_gray = axs[i][2]
        show_psd(
            ax_gray,
            psd_th_sub[i].astype(float),
            fpsd,
            tpsd_sub,
            vmin=0,
            vmax=1,
            cmap="gray",
            interpolation="none",
        )
        set_ticks(ax_gray, tl, yl)

        if i == 1:
            ax_psd.set_xlabel("Time (s)")
            ax_gray.set_xlabel("Time (s)")

        x0 = tl[0] + (tl[1] - tl[0]) / 12
        y0 = yl[1] - (yl[1] - yl[0]) / 12
        for j, ax in enumerate((ax_psd, ax_gray)):
            ax.text(
                x0, y0, lb_set[i], fontsize=7.5, color=color_set[j], va="top", ha="left"
            )

    axs[0][0].set_title("Power spectrogram")
    axs[0][2].set_title("Quantized\npower spectrogram")

    return fig


@fm.figure_renderer("comap_sample", reset=False)
def draw_comap_sample(
    figsize=(4.2, 4), cid=7, fdir_coburst="", vmax=0.02, cmap="turbo"
):

    comap = xa.load_dataarray(os.path.join(fdir_coburst, "co_map_%d.nc" % (cid)))
    fpsd = comap.coords["f1"]
    im_comap = comap.sel(dict(mv="mean", type="fs")).values.copy()

    fig = uf.get_figure(figsize)
    ax = plt.axes(position=(0.05, 0.05, 0.9, 0.9))
    plt.imshow(
        im_comap,
        cmap=cmap,
        extent=(fpsd[0], fpsd[-1], fpsd[0], fpsd[-1]),
        origin="lower",
        vmin=0,
        vmax=vmax,
    )

    plt.xlim([20, 80])
    plt.ylim([20, 80])
    plt.xticks(np.arange(20, 81, 10))
    plt.yticks(np.arange(20, 81, 10))
    plt.ylabel(r"Frequency of Pop$_{F}$ (Hz)")
    plt.xlabel(r"Frequency of Pop$_{S}$ (Hz)")
    plt.title("Burst co-occurrence\nprobability")

    return fig


@fm.figure_renderer("comap_colorbar", reset=False)
def draw_comap_colorbar(figsize=(4.2, 4), vmax=0.02, cmap="turbo"):

    fig1 = plt.figure()
    im = plt.imshow(np.linspace(0, vmax, 9).reshape(3, 3), cmap=cmap)

    fig = uf.get_figure(figsize)
    ax = plt.axes(position=(0.05, 0.05, 0.9, 0.9))
    plt.colorbar(im, cax=ax, ticks=np.linspace(0, vmax, 3))
    fig1.clf()

    return fig


if __name__ == "__main__":
    cid = 7
    im_opt = dict(vmax=0.02, cmap="turbo")
    h = 3.5

    draw_example(figsize=(6, h), fdir_simul=fdir_simul, cid=cid, nt=93)
    draw_comap_sample(figsize=(h, h), cid=cid, fdir_coburst=fdir_coburst, **im_opt)
