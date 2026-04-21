import numpy as np
import matplotlib.pyplot as plt
import figure_manager as fm
import yaml
import os

from pytools import hhsignal, visu
from pytools import oscdetector as od
from pytools import utils_te as ut
from pytools import utils_fig as uf

uf.set_plt()

empty_files = False
te_colors = ("#d45b0c", "#003f5c", "#a9a9a9")
c_circ = "#7d0000"
c_rect = "#676767"
te_dir = "../results/te/"

fm.track_global("te_colors", te_colors)
fm.track_global("te_dir", te_dir)
fm.track_global("c_rect", c_rect)


with open("./cw_pos.yaml", "r") as fp:
    cw_id_pairs = yaml.safe_load(fp)["cwpos"]


def gen_signal(tmax, t_pts, f):
    srate = 2000
    t = np.arange(0, tmax, 1 / srate)
    y = np.zeros_like(t)

    s = 0.2
    for t0 in t_pts:
        assert t0 < tmax
        n0 = int(t0 * srate)
        nw = int(10 * s * srate)

        tsub = np.arange(-nw, nw + 1) / srate
        ysub = np.cos(2 * np.pi * f * tsub) * np.exp(-(tsub**2) / s) * 0.5

        y[n0 - nw : n0 + nw + 1] += ysub

    return t, y


def show_schem_spec(psd, tpsd, fpsd, yl=None, pop_txt="Fast", cmap="jet"):
    vmin, vmax = None, None
    if "RdBu" in cmap:

        vmin, vmax = -0.2, 0.2

    plt.imshow(
        psd,
        extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.xticks([])
    plt.yticks([])

    plt.ylim(yl)
    plt.ylabel("Frequency (Hz)", fontsize=5)

    xl = plt.xlim()
    yl = plt.ylim()

    x0 = xl[0] + 0.15 * (xl[1] - xl[0])
    y0 = yl[1] - 0.15 * (yl[1] - yl[0])

    plt.text(
        x0,
        y0,
        pop_txt,
        color="w",
        ha="center",
        va="center",
        fontsize=6,
        fontweight="bold",
    )


def get_psd_set(detail):
    psd_set = []
    for i in range(2):
        psd, fpsd, tpsd = hhsignal.get_stfft(
            detail["vlfp"][i + 1], detail["ts"], 2000, frange=(5, 110)
        )
        psd_set.append(psd)
    return psd_set, fpsd, tpsd


def show_psd(psd, fpsd, tpsd, vmin=0, vmax=1):
    plt.imshow(
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


def set_ticks(tl):
    plt.xlim(tl)
    plt.xticks(np.arange(tl[0], tl[1] + 1e-3, 0.5))
    plt.yticks(np.arange(20, 101, 20))
    plt.ylim([18, 82])
    plt.gca().set_xticklabels([])


def set_colorbar(cticks=None):
    cbar = plt.colorbar()
    cbar.set_ticks(cticks)


@fm.figure_renderer("te_example", reset=empty_files)
def draw_example_te(figsize=(2.8, 11.8), cid=5, wid=10, p_ranges=(5, 95), te_dir=None):

    tcut = ut.get_max_period(cid)
    te_data_2d = uf.load_pickle(os.path.join(te_dir, "te_%d%02d.pkl" % (cid, wid)))
    te_data = ut.reduce_te_2d(te_data_2d, tcut=tcut)

    tlag = te_data["tlag"]

    lb_pop = ("F", "S")
    ybar = 0.065
    fig = uf.get_figure(figsize)

    id_sig_sets = ut.identify_sig_te1d(te_data, prt=p_ranges[1])
    tsig_sets = ut.convert_sig_boundary(id_sig_sets, tlag)

    shown_sig_process = False

    for nd in range(2):

        plt.axes([0.12, 0.47 + (1 - nd) * 0.27, 0.8, 0.19])

        x1 = te_data["te"][:, nd, :]
        x2 = te_data["te_surr"][:, nd, :]
        tlag = te_data["tlag"]

        opt = dict(alpha=0.5, avg_method="median", p_range=p_ranges)
        opt_line = dict(linestyle="-", linewidth=0.5)
        opt_noline = dict(linestyle="none")
        visu.draw_with_err(tlag, x1, c=te_colors[nd], **opt, **opt_noline)

        visu.draw_with_err(tlag, x2, c=te_colors[2], **opt, **opt_noline)
        (l1,) = plt.plot(tlag, np.median(x1, axis=0), c=te_colors[nd], **opt_line)
        (l2,) = plt.plot(tlag, np.median(x2, axis=0), c=te_colors[2], **opt_line)

        for tsig in tsig_sets[nd]:
            plt.plot(tsig, [ybar] * 2, "k-", lw=0.5, markersize=3)

            if not shown_sig_process:
                ytrue = np.median(x1, axis=0)
                ysurr = np.percentile(x2, p_ranges[1], axis=0)
                n0 = np.where(tlag == tsig[0])[0][0]
                n1 = np.where(tlag == tsig[1])[0][0]

                if ysurr[n0] > ysurr[n1]:
                    nth, nth_id = n0, 0
                else:
                    nth, nth_id = n1, 1

                opt_sub = dict(color="k", linestyle="--", linewidth=0.5)
                plt.plot(tsig, [ytrue[nth], ytrue[nth]], **opt_sub)
                plt.plot([tsig[0]] * 2, [ytrue[nth], ybar], **opt_sub)
                plt.plot([tsig[1]] * 2, [ytrue[nth], ybar], **opt_sub)

                nmax = np.argmax(ytrue[n0:n1]) + n0
                plt.plot(tlag[nmax], ybar, "k*", markersize=3)
                plt.plot(tsig[nth_id], ytrue[nth], "kx", markersize=3)

        plt.xlim([1, tcut])
        plt.ylim([-0.001, 0.08])

        plt.xlabel(r"$\tau$ (ms)")
        plt.ylabel(r"$TE_{%s \rightarrow %s}$ (bits)" % (lb_pop[nd], lb_pop[1 - nd]))

        plt.legend(
            [l1, l2],
            ("TE", r"TE$^{surr}$"),
            fontsize=5,
            loc="upper right",
            edgecolor="none",
            facecolor="none",
            borderpad=0,
            borderaxespad=0,
            handlelength=1,
            handletextpad=0.8,
            labelspacing=0.25,
        )

    plt.axes([0.01, 0.22, 0.93, 0.15])
    visu.draw_te_diagram_full(
        tsig_sets, xmax=tcut, y0=30, colors_arrow=[c_rect] * 2, colors_rect=[c_rect] * 2
    )

    box_height = 2
    plt.axes([0.01, 0.05, 0.93, 0.12])
    visu.draw_te_diagram_reduce(
        tsig_sets,
        xmax=tcut,
        y0=2 * box_height,
        colors=[c_rect] * 2,
        box_height=box_height,
        visu_type="arrow",
    )

    return fig


@fm.figure_renderer("entire_irp", reset=empty_files)
def draw_entire_irp(figsize=(8.5, 9.8), p_ranges=(5, 95)):

    num_row = 7
    num_col = 3

    ws_row = 0.002
    ws_col = 0.02

    wr = (1 - (num_row + 1) * ws_row) / num_row
    wc = (1 - (num_col + 1) * ws_col) / num_col
    box_height = 2

    fig = uf.get_figure(figsize=figsize)
    k = 0
    for nr in range(num_row):
        for nc in range(len(cw_id_pairs[nr])):
            lid, cw = cw_id_pairs[nr][nc]
            if len(cw) == 0:
                continue

            cid, wid = cw
            max_period = ut.get_max_period(cid)

            te_data_2d = uf.load_pickle(
                os.path.join(te_dir, "te_%d%02d.pkl" % (cid, wid))
            )
            te_data = ut.reduce_te_2d(te_data_2d, tcut=max_period)
            tlag = te_data["tlag"]

            id_sig_sets = ut.identify_sig_te1d(te_data, prt=p_ranges[1])
            tsig_sets = ut.convert_sig_boundary(id_sig_sets, tlag)

            pos = (
                (nc + 1) * ws_col + nc * wc,
                1 - ((nr + 1) * ws_row + nr * wr) - wr,
                wc,
                wr,
            )
            ax = plt.axes(pos)

            ax_te = ax.inset_axes([0.23, 0.0, 0.77, 1])
            fig.add_axes(ax_te)
            plt.sca(ax_te)

            box_height = 2
            visu.draw_te_diagram_reduce(
                tsig_sets,
                xmax=max_period,
                y0=2 * box_height,
                colors=[c_rect] * 2,
                box_height=box_height,
                visu_type="arrow",
                fontsize=6,
            )

            ax_pict = ax.inset_axes([0.05, 0.1, 0.2, 0.8])
            fig.add_axes(ax_pict)
            plt.sca(ax_pict)
            uf.draw_motif_pictogram(
                od.get_motif_labels()[wid], rcolor=uf.get_cid_color(cid)
            )

            ax.axis("off")

            ax.text(-1, 1, "%d" % (lid), fontsize=6, ha="center", va="center")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1.2, 1.2])

            k += 1

    return fig


def gen_sample1(T=1000):
    tau, s = 9, 3

    X = np.zeros(T)
    for t in range(1, T):
        X[t] = 0.6 * X[t - 1] + 0.4 * np.random.randn()

    Y = np.zeros(T)
    for t in range(T):
        y_term = 0.1 * Y[t - s] if t - s >= 0 else 0.0

        x_term = 0.5 * X[t - tau] if t - tau >= 0 else 0.0
        Y[t] = y_term + 0.2 * np.random.randn()

    valid = np.arange(max(tau, s), T)
    Y_t = Y[valid]
    X_tau = X[valid - tau]
    Y_s = Y[valid - s]

    return Y_t, Y_s, X_tau


def gen_sample2(T=1000):
    tau, s = 9, 3

    X = np.zeros(T)
    for t in range(1, T):
        X[t] = 0.8 * X[t - 1] + 0.2 * np.random.randn()

    Y = np.zeros(T)
    for t in range(T):
        y_term = 0.1 * Y[t - s] + 0.5 * X[t - tau] if t - s >= 0 else 0.0
        x_term = 0.6 * X[t - tau] if t - tau >= 0 else 0.0

        Y[t] = y_term + 2 * x_term + 0.2 * np.random.randn()

    valid = np.arange(max(tau, s), T)
    Y_t = Y[valid]
    X_tau = X[valid - tau]
    Y_s = Y[valid - s]

    return Y_t, Y_s, X_tau


def gen_sample3(T=1000):
    tau, s = 9, 3

    X = np.zeros(T)
    for t in range(1, T):
        X[t] = 0.8 * X[t - 1] + 0.2 * np.random.randn()

    Y = np.zeros(T)
    for t in range(T):
        y_term = 0.2 * Y[t - s] if t - s >= 0 else 0.0
        x_term = 0.5 * X[t - tau] if t - tau >= 0 else 0.0
        Y[t] = y_term + 0.2 * x_term + 0.2 * np.random.randn()

    valid = np.arange(max(tau, s), T)
    Y_t = Y[valid]
    X_tau = X[valid - tau]
    Y_s = Y[valid - s]

    return Y_t, Y_s, X_tau


def draw_hist(ax, Y_t, Y_s, X_tau):
    bins = 21
    data = np.vstack([Y_t, X_tau, Y_s]).T
    H, edges = np.histogramdd(data, bins=bins)

    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    cy, cx, cz = np.meshgrid(centers[0], centers[1], centers[2], indexing="ij")

    mask = H > 0
    xs = cx[mask].ravel()
    ys = cy[mask].ravel()
    zs = cz[mask].ravel()
    cs = H[mask].ravel()

    cs_norm = (cs - cs.min()) / (cs.max() - cs.min() + 1e-12)

    sizes = 0.5 + 0.2 * cs_norm
    alphas = 0.2 + 0.7 * cs_norm

    pts = ax.scatter(ys, xs, zs, s=sizes, c="k", alpha=1.0, edgecolors="none", zorder=0)

    pts.set_facecolors([[0, 0, 0, a] for a in alphas])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        for line in axis.get_ticklines():
            line.set_linewidth(0.2)
        for line in axis.get_ticklines(minor=True):
            line.set_linewidth(0.2)

    ax.xaxis.line.set_linewidth(0.2)
    ax.yaxis.line.set_linewidth(0.2)
    ax.zaxis.line.set_linewidth(0.2)


@fm.figure_renderer("irp_4dhist_reduce", reset=empty_files, exts=[".png", ".svg"])
def irp_4dhist_reduce(figsize=(2, 4)):

    np.random.seed(42)

    T = 300

    fig = uf.get_figure(figsize)
    axs = []
    for n in range(3):
        axs.append(fig.add_subplot(3, 1, n + 1, projection="3d"))

    draw_hist(axs[0], *gen_sample1(T=T))
    draw_hist(axs[1], *gen_sample2(T=T))
    draw_hist(axs[2], *gen_sample3(T=T))

    return fig


def main():
    p_ranges = (2.5, 97.5)

    draw_example_te(
        te_dir=te_dir,
        cid=4,
        wid=10,
        p_ranges=p_ranges,
        _func_label="draw_example_te_410",
        _transparent=True,
    )
    draw_example_te(
        te_dir=te_dir,
        cid=4,
        wid=15,
        p_ranges=p_ranges,
        _func_label="draw_example_te_415",
        _transparent=True,
    )
    draw_entire_irp(figsize=(9.5, 11), p_ranges=p_ranges, _transparent=True)
    irp_4dhist_reduce(_transparent=True)


if __name__ == "__main__":
    main()
