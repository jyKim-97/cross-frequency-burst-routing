import numpy as np
import matplotlib.pyplot as plt
import xarray as xa
from warnings import warn
import os
import pickle as pkl

from pytools import hhsignal, hhtools
from pytools import utils_fig as uf

uf.set_plt()

import figure_manager as fm

file_umap = "../results/twopop_output/umap_coord.nc"
file_postdata = "../results/twopop_output/processed_results.nc"
fdir_bprops = "../results/twopop_regime_samples/bprops"
file_repr = "../results/clustering/cluster_landmarks.pkl"
file_cluster = "../results/clustering/cluster_id.nc"

fdir_simul = "../results/twopop_regime_samples/data"
fdir_simul_example = "../results/twopop_regime_samples"


fm.track_global("file_umap", file_umap)
fm.track_global("file_postdata", file_postdata)
fm.track_global("fdir_bprops", fdir_bprops)
fm.track_global("fdir_simul", fdir_simul)
fm.track_global("file_repr", file_repr)
fm.track_global("file_cluster", file_cluster)

reset = False


def get_psd_set(detail):
    psd_set = []
    for i in range(2):
        psd, fpsd, tpsd = hhsignal.get_stfft(
            detail["vlfp"][i + 1], detail["ts"], 2000, frange=(5, 110)
        )
        psd_set.append(psd)

    return psd_set, fpsd, tpsd


def show_psd(psd, fpsd, tpsd, vmin=0.0, vmax=1.0, tl=(0, 1)):
    im_obj = plt.imshow(
        psd,
        aspect="auto",
        cmap="jet",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
        interpolation="bicubic",
    )
    plt.ylabel("Frequency (Hz)")
    x0 = tl[0] + 1
    uf.show_scalebar(
        plt.gca(),
        size=1,
        label="1 s",
        anchor_pos=(x0, 25),
        lw=1,
        pad=5,
        color="w",
        fontsize=5,
    )
    set_ticks(tl)

    return im_obj


def hist2d(y, x, yedges, xedges):
    idx = np.digitize(x, xedges, right=False)
    idy = np.digitize(y, yedges, right=False)

    id_nan = (idx == 0) | (idx == len(xedges))
    id_nan = id_nan | (idy == 0)
    id_nan = id_nan | (idy == len(yedges))

    idx = idx[~id_nan] - 1
    idy = idy[~id_nan] - 1

    matsize = (len(yedges) - 1, len(xedges) - 1)
    num_hist = _count(matsize, idx, idy)
    return num_hist


def get_mid_pts(edges):
    return (edges[1:] + edges[:-1]) / 2


def _count(matsize, idx, idy):
    num = np.zeros(matsize)
    for n in range(len(idx)):
        num[idy[n], idx[n]] += 1
    return num


def draw_burst_props(bprop_f, bprop_l, nbins=21, vmax=0.05, tl=(0, 1)):
    from scipy.ndimage import gaussian_filter

    fedges = np.arange(5, 101, 5)
    ledges = np.linspace(-0.2, 0.6, 21)

    y = get_mid_pts(fedges)
    x = get_mid_pts(ledges)

    im = hist2d(bprop_f, bprop_l, fedges, ledges)
    im = gaussian_filter(im, 0.8)

    im_obj = plt.contourf(
        x,
        y,
        im / im.sum(),
        np.concatenate((np.linspace(0, vmax, nbins), [1])),
        cmap="turbo",
        vmax=vmax,
        vmin=0,
    )

    plt.ylim([10, 90])
    plt.yticks(np.arange(10, 91, 20))

    plt.xlabel("Burst duration (s)", labelpad=1.5)
    plt.ylabel("Frequency (Hz)")
    plt.xticks(np.linspace(tl[0], tl[1], 3))
    plt.xlim(tl)

    return im_obj


def set_ticks(tl):
    plt.xlim(tl)
    plt.xticks(np.arange(tl[0], tl[1] + 1e-3))
    plt.yticks(np.arange(10, 91, 20))
    plt.ylim([10, 90])
    plt.gca().set_xticklabels([])


def show_label(lb):
    xl = plt.xlim()
    yl = plt.ylim()
    x0 = xl[1] - (xl[1] - xl[0]) / 6
    y0 = yl[1] - (yl[1] - yl[0]) / 7
    plt.text(x0, y0, lb, fontsize=6, va="center", ha="center", color="w")


def set_colorbar(im_obj, ax, cticks=None):
    cbar = plt.colorbar(im_obj, cax=ax, shrink=1)
    cbar.set_ticks(cticks)


@fm.figure_renderer("landmark_prop", reset=False)
def show_landmark_prop(
    figsize=(2.35, 9.5),
    cid=0,
    nt=0,
    vmins_spec=(0.1, 0.1),
    vmaxs_spec=(0.6, 1),
    tl_spec=(1, 9),
    tl_burst=(0, 0.5),
    vmax_burst=0.04,
):

    fig = uf.get_figure(figsize)
    axes = uf.get_custom_subplots(
        h_ratio=[0.8, 1, 1, 1, 1],
        w_ratio=[1, 0.03],
        h_blank_interval_set=[0.04, 0.04, 0.09, 0.07],
        w_blank_interval_set=[0.05],
        h_blank_boundary=0.02,
        w_blank_boundary=0.05,
    )

    plt.sca(axes[0][0])
    uf.draw_landmark_diagram(cid)
    axes[0][1].axis("off")
    labels = ("Fast", "Slow")

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

    for ntp in range(2):
        plt.sca(axes[ntp + 1][0])
        vmin, vmax = vmins_spec[ntp], vmaxs_spec[ntp]
        im_spec = show_psd(psd_set[ntp], fpsd, tpsd, vmin=vmin, vmax=vmax, tl=tl_spec)
        set_colorbar(im_spec, ax=axes[ntp + 1][1], cticks=np.linspace(vmin, vmax, 3))
        show_label(labels[ntp])

    axes[1][0].set_title("Power spectrum")

    bprop_set = uf.load_pickle(os.path.join(fdir_bprops, "bprops_%d.pkl" % (cid)))

    mbin_t = bprop_set["attrs"]["mbin_t"]

    for ntp in range(2):
        plt.sca(axes[ntp + 3][0])
        b = bprop_set["burst_props"][ntp]
        blen = np.array(b["burst_range"][:, 1] - b["burst_range"][:, 0]) * mbin_t
        im_bp = draw_burst_props(
            bprop_set["burst_props"][ntp]["burst_f"],
            blen,
            nbins=31,
            vmax=vmax_burst,
            tl=tl_burst,
        )
        set_colorbar(im_bp, ax=axes[ntp + 3][1], cticks=np.linspace(0, vmax_burst, 3))
        show_label(labels[ntp])

        axes[ntp + 3][1].set_ylim([0, vmax_burst])

    axes[3][0].set_title("Burst feature density")

    return fig


def sel_params(postdata, key=None, pop=None, type=None):
    return postdata.sel(dict(key=key, pop=pop, type=type)).data.flatten()


def draw_result(
    params,
    figsize=(5, 5),
    title=None,
    s=1.0,
    edgecolor="none",
    vmin=None,
    vmax=None,
    cmap="viridis",
    shrink=0.5,
    cticks=None,
    **plot_opt,
):

    postdata = xa.load_dataarray(file_postdata)
    umap_coord = xa.load_dataarray(file_umap)

    d = sel_params(postdata, **params)
    if "tlag" in params["key"]:
        d = 1 / d

    fig = uf.get_figure(figsize)
    plt.scatter(
        umap_coord[:, 0],
        umap_coord[:, 1],
        s=s,
        edgecolor=edgecolor,
        cmap=cmap,
        c=d,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        **plot_opt,
    )
    plt.colorbar(shrink=shrink, ticks=cticks)

    MARKERS = ("o", "^", "x")

    da_zeros = umap_coord[umap_coord.zero_idx]

    for i in range(3):
        plt.scatter(da_zeros[i, 0], da_zeros[i, 1], s=s, c="k", marker=MARKERS[i])

    plt.axis("off")
    if title is None:
        title = params["key"]
    plt.title(title, fontsize=12)

    return fig


@fm.figure_renderer("chi")
def draw_params_on_umap(
    key: str = "chi",
    pop: str = "F",
    lb_title: str = "none",
    vmin=0,
    vmax=0.4,
    cticks=(0, 0.2, 0.4),
    s=0.5,
    cmap="viridis",
    figsize=(5, 5),
):

    params = dict(key=key, pop=pop, type="mean")
    fig = draw_result(
        params,
        figsize=figsize,
        title=lb_title,
        s=s,
        cticks=cticks,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    return fig


@fm.figure_renderer("umap")
def draw_cluster_umap(figsize=(5, 5)):
    cdata = xa.load_dataset(file_cluster)
    cid = cdata.cluster_id.data.flatten()
    repr_idx = uf.load_pickle(file_repr)

    num_cluster = 7
    nshape = (15, 15, 3, 16)

    idx = []
    for i in range(num_cluster):
        nid_set = repr_idx["repr_idx"][i]
        nid = (
            nid_set[-1]
            + nid_set[-2] * nshape[-1]
            + nid_set[-3] * nshape[-1] * nshape[-2]
            + nid_set[0] * nshape[1] * nshape[2] * nshape[3]
        )
        idx.append(nid)

    umap_coord = xa.load_dataarray(file_umap)

    fig = uf.get_figure(figsize)

    cmap = plt.get_cmap("turbo", len(np.unique(cid)))
    plt.scatter(
        umap_coord[:, 0],
        umap_coord[:, 1],
        s=0.5,
        edgecolor="none",
        c=cid,
        cmap=cmap,
        rasterized=True,
        alpha=0.6,
    )
    plt.axis("off")
    cbar = plt.colorbar(shrink=0.5, ticks=np.arange(1 + 3 / 7, 7 - 3 / 7 + 1e-3, 6 / 7))
    cbar.set_ticklabels(["%d" % (n) for n in range(1, 8)])

    for i in range(num_cluster):

        plt.scatter(
            umap_coord[idx[i], 0],
            umap_coord[idx[i], 1],
            marker="o",
            s=10,
            edgecolor="k",
            color=cmap(i / num_cluster),
        )

    return fig


if __name__ == "__main__":
    _transparent = True

    draw_params_on_umap(
        key="chi",
        pop="F",
        lb_title=r"$chi^{Fast}_1$",
        _func_label="echelon_fast",
        _transparent=_transparent,
    )
    draw_params_on_umap(
        key="chi",
        pop="S",
        lb_title=r"$chi^{Slow}_1$",
        _func_label="echelon_slow",
        _transparent=_transparent,
    )

    opt_freq = dict(
        vmin=30, vmax=70, cticks=(30, 50, 70), cmap="RdBu_r", _transparent=_transparent
    )
    draw_params_on_umap(
        key="tlag_1st",
        pop="F",
        lb_title=r"$f^{Fast}_1$",
        **opt_freq,
        _func_label="freq_fast_1",
    )
    draw_params_on_umap(
        key="tlag_large",
        pop="F",
        lb_title=r"$f^{Fast}_M$",
        **opt_freq,
        _func_label="freq_fast_M",
    )
    draw_params_on_umap(
        key="tlag_1st",
        pop="S",
        lb_title=r"$f^{Slow}_1$",
        **opt_freq,
        _func_label="freq_slow_1",
    )
    draw_params_on_umap(
        key="tlag_large",
        pop="S",
        lb_title=r"$f^{Slow}_M$",
        **opt_freq,
        _func_label="freq_slow_M",
    )

    show_landmark_prop(
        cid=7,
        vmins_spec=(0.1, 0.1),
        vmaxs_spec=(0.6, 0.8),
        _func_label="landmark_prop_7",
        _transparent=_transparent,
    )
    show_landmark_prop(
        cid=5,
        vmins_spec=(0.1, 0.1),
        vmaxs_spec=(0.8, 0.4),
        _func_label="landmark_prop_5",
        _transparent=_transparent,
    )
    show_landmark_prop(
        cid=4,
        vmins_spec=(0.1, 0.1),
        vmaxs_spec=(0.4, 0.6),
        _func_label="landmark_prop_4",
        _transparent=_transparent,
    )

    draw_cluster_umap()
