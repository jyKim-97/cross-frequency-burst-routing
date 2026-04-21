import numpy as np
import matplotlib.pyplot as plt
import figure_manager as fm
import yaml
import os
import xarray as xa
from scipy.ndimage import gaussian_filter1d

from pytools import visu
from pytools import oscdetector as od
from pytools import utils_te as ut
from pytools import utils_fig as uf

uf.set_plt()

empty_files = False

prob_spk_dir = "../results/spike_transmission/recv_probs"
kappa_dir = "../results/spike_transmission/kappa"

tl_colors = ("#d45b0c", "#003f5c", "#a9a9a9")
arrow_colors = ("#d92a27", "#045894", "#a9a9a9")
c_rect = "#676767"
box_height = 2

fm.track_global("tl_colors", tl_colors)
fm.track_global("arrow_colors", arrow_colors)
fm.track_global("kappa_dir", kappa_dir)


with open("./cw_pos.yaml", "r") as fp:
    cw_id_pairs = yaml.safe_load(fp)["cwpos"]


def load_kappa(kappa_dir, cid, wid):
    fname = os.path.join(kappa_dir, "kappa_%d%02d.nc" % (cid, wid))
    return xa.open_dataset(fname)


def get_err_range(data, method="percentile", p_ranges=(5, 95), smul=1.96):

    if method == "percentile":
        m = np.median(data, axis=0)
        smin = np.percentile(data, p_ranges[0], axis=0)
        smax = np.percentile(data, p_ranges[1], axis=0)
    elif method == "std":
        m = data.mean(axis=0)
        s = data.std(axis=0) / np.sqrt(data.shape[0]) * smul
        smin = m - s
        smax = m + s
    else:
        raise ValueError("Unknown method: %s" % method)

    return m, smin, smax


@fm.figure_renderer("single_resp_sample", reset=empty_files)
def show_single_spike_resp(
    figsize=(2.8, 2.2), prob_spk_dir=None, cid=0, wid=10, ntp=1, s=1.5
):
    lb = "F" if ntp == 0 else "S"

    prob_set = xa.load_dataset(os.path.join(prob_spk_dir, "prob_spk%d.nc" % (cid)))
    prob_set = prob_set.sel(dict(nw=wid))

    t = prob_set.t

    cmap = plt.get_cmap("turbo")

    fig = uf.get_figure(figsize)

    for nd in prob_set.ndelay:
        prob_sub = prob_set.sel(dict(ndelay=nd, ntp=ntp))
        p = prob_sub.prob.data
        p0 = prob_sub.prob0.data

        dy = p.mean(axis=0) - p0.mean(axis=0)
        dy = gaussian_filter1d(dy, s)

        ds = p.std(axis=0) / np.sqrt(p.shape[0]) * 0.98

        plt.plot(t, dy, c=cmap(nd / prob_set.ndelay.max()))
        plt.fill_between(
            t,
            dy - ds,
            dy + ds,
            color=cmap(nd / prob_set.ndelay.max()),
            alpha=0.3,
            edgecolor="none",
        )

    plt.xlabel("Time from" + "\n" + "transmitter firing (ms)")

    plt.ylabel(r"$\Delta p_{%s} = p_{%s} - p_{%s}^{base}$" % (lb, lb, lb))
    plt.ylim([-0.001, 0.02])

    return fig


@fm.figure_renderer("show_tline_sample", reset=empty_files)
def show_tline_sample(
    figsize=(2.8, 7.8),
    kappa_dir=None,
    cid=0,
    wid=0,
    err_method="std",
    err_std=1.96,
    p_ranges=(5, 95),
):
    lb_set = ("S", "F")

    kappa_set = load_kappa(kappa_dir, cid, wid)
    tmax = ut.get_max_period(cid)

    dtq = 0.5
    id_sig_pos, id_sig_neg, tq = ut.identify_sig_tline(
        kappa_set,
        err_method=err_method,
        err_std=err_std,
        p_ranges=p_ranges,
        num_min=4,
        dtq=dtq,
    )
    tsig_pos_set = ut.convert_sig_boundary(id_sig_pos, tq)
    tsig_neg_set = ut.convert_sig_boundary(id_sig_neg, tq)

    ybar = 0.23
    ylim = 0.25

    fig = uf.get_figure(figsize)

    axs = []
    for npop in range(2):
        ax = plt.axes([0.12, 0.7 - 0.35 * npop, 0.8, 0.25])
        axs.append(ax)

        t = kappa_set.ndelay.data
        ym_b, ymin_b, ymax_b = get_err_range(
            kappa_set.kappa_base.isel(dict(ntp=1 - npop)).data,
            method=err_method,
            smul=err_std,
        )
        ym, ymin, ymax = get_err_range(
            kappa_set.kappa.isel(dict(ntp=1 - npop)).data,
            method=err_method,
            smul=err_std,
        )

        plt.plot(t, ym, color=tl_colors[npop], lw=1, label=r"$\kappa$")
        plt.fill_between(
            t, ymin, ymax, color=tl_colors[npop], alpha=0.3, edgecolor="none"
        )
        plt.fill_between(
            t,
            ymin_b,
            ymax_b,
            color="k",
            alpha=0.5,
            edgecolor="none",
            label=r"$\kappa^{out}$",
        )
        plt.legend(
            loc="lower right", fontsize=4.5, edgecolor="none", facecolor="none", ncol=2
        )

        ymin_b = np.interp(tq, t, ymin_b)
        ymax_b = np.interp(tq, t, ymax_b)
        ymin = np.interp(tq, t, ymin)
        ymax = np.interp(tq, t, ymax)
        ym = np.interp(tq, t, ym)

        opt_sub = dict(color="k", linestyle="--", lw=0.5)
        for nd, tsig_set in enumerate([tsig_pos_set, tsig_neg_set]):
            y0 = (-1) ** nd * ybar
            if nd == 0:
                ytrue, yout = ymin, ymax_b
            else:
                ytrue, yout = ymax, ymin_b

            for tsig in tsig_set[npop]:
                n0 = np.where(tq == tsig[0])[0][0]
                n1 = np.where(tq == tsig[1])[0][0]

                if yout[n0] * (-1) ** nd > yout[n1] * (-1) ** nd:
                    nth, nth_id = n0, 0
                else:
                    nth, nth_id = n1, 1

                plt.plot(tsig, [y0] * 2, "k-", lw=0.5)
                plt.plot([tsig[0]] * 2, [ytrue[nth], y0], **opt_sub)
                plt.plot([tsig[1]] * 2, [ytrue[nth], y0], **opt_sub)

                nmax = np.argmax(ytrue[n0:n1] * (-1) ** nd) + n0
                plt.plot(tq[nmax], y0, "k*", markersize=3)
                plt.plot(tsig[nth_id], ytrue[nth], "kx", markersize=3)

        plt.xticks(np.arange(0, 31, 10))
        plt.yticks(
            np.arange(-ylim, ylim + 0.1, 0.1),
            labels=[
                str("%d %%" % (int(100 * x))) for x in np.arange(-ylim, ylim + 0.1, 0.1)
            ],
        )
        plt.ylim([-ylim - 0.02, ylim + 0.02])
        plt.xlim([0, tmax])
        plt.xlabel("Delay, d (ms)")
        plt.ylabel(r"$\kappa_{%s \rightarrow %s}$" % (lb_set[1 - npop], lb_set[npop]))

    ax = plt.axes([0.01, 0.05, 0.95, 0.2])

    box_height = 2
    opt_bbox = dict(
        y0=2 * box_height,
        box_height=box_height,
        visu_type="arrow",
        alpha=0.8,
        show_axis=False,
    )
    visu.draw_te_diagram_reduce(
        tsig_pos_set, tmax, colors=[arrow_colors[0]] * 2, **opt_bbox
    )
    visu.draw_te_diagram_reduce(
        tsig_neg_set, tmax, colors=[arrow_colors[1]] * 2, **opt_bbox
    )
    visu.draw_te_diagram_reduce(
        [[], []],
        colors=[c_rect] * 2,
        xmax=tmax,
        y0=2 * box_height,
        box_height=box_height,
        visu_type="box",
        show_axis=True,
        fontsize=6,
    )

    ax.axis("off")

    return fig


def _span_bbox(r0, r1, c0, c1, wr, wc, ws_row, ws_col):

    x = (c0 + 1) * ws_col + c0 * wc
    y = 1 - ((r1 + 1) * ws_row + r1 * wr) - wr
    w = (c1 - c0 + 1) * wc + (c1 - c0) * ws_col
    h = (r1 - r0 + 1) * wr + (r1 - r0) * ws_row
    return [x, y, w, h]


@fm.figure_renderer("delay_colorbar")
def draw_cbar(figsize=(0.25, 2)):

    fig = uf.get_figure(figsize)
    im = np.arange(0, 31, 2).reshape(-1, 1)
    plt.imshow(
        im,
        cmap="turbo",
        vmin=0,
        vmax=30,
        aspect="auto",
        origin="lower",
        extent=(0, 1, -1, 31),
    )
    plt.gca().yaxis.tick_right()
    plt.xticks([])
    plt.yticks(np.arange(0, 31, 10))
    return fig


@fm.figure_renderer("draw_entire_irp", reset=empty_files)
def draw_entire_tl(figsize=(9.5, 11), kappa_dir=None, err_std=2.58, p_ranges=(5, 95)):

    num_row = 7
    num_col = 3

    ws_row = 0.002
    ws_col = 0.02

    wr = (1 - (num_row + 1) * ws_row) / num_row
    wc = (1 - (num_col + 1) * ws_col) / num_col

    fig = uf.get_figure(figsize)

    k = 0
    for nr in range(num_row):
        for nc in range(len(cw_id_pairs[nr])):
            lid, cw = cw_id_pairs[nr][nc]
            if len(cw) == 0:
                continue

            cid, wid = cw
            max_period = ut.get_max_period(cid)

            kappa_set = load_kappa(kappa_dir, cid, wid)
            id_sig_pos, id_sig_neg, tq = ut.identify_sig_tline(
                kappa_set,
                err_method="std",
                err_std=err_std,
                p_ranges=p_ranges,
                num_min=4,
            )
            tline_sig_pos = ut.convert_sig_boundary(id_sig_pos, tq)
            tline_sig_neg = ut.convert_sig_boundary(id_sig_neg, tq)

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

            box_opt = dict(xmax=max_period, y0=2 * box_height, box_height=box_height)

            tline_opt = dict(visu_type="arrow", show_axis=False, alpha=0.8)
            visu.draw_te_diagram_reduce(
                tline_sig_pos,
                colors=[arrow_colors[0]] * 2,
                y0_set=[0, 5 - box_height / 2],
                xmax=max_period,
                box_height=box_height,
                **tline_opt,
            )
            visu.draw_te_diagram_reduce(
                tline_sig_neg,
                colors=[arrow_colors[1]] * 2,
                y0_set=[0, 5 - box_height / 2],
                xmax=max_period,
                box_height=box_height,
                **tline_opt,
            )

            visu.draw_te_diagram_reduce(
                [[], []],
                colors=[c_rect] * 2,
                **box_opt,
                visu_type="box",
                show_axis=True,
                fontsize=6,
            )

            ax_pict = ax.inset_axes([0.05, 0.1, 0.2, 0.8])
            fig.add_axes(ax_pict)
            plt.sca(ax_pict)
            uf.draw_motif_pictogram(
                od.get_motif_labels()[wid],
                rcolor=uf.get_cid_color(cid),
            )

            ax.axis("off")
            ax.text(-1, 1, "%d" % (lid), fontsize=6, ha="center", va="center")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1.2, 1.2])

            k += 1

    return fig


def main():
    draw_cbar()
    show_single_spike_resp(
        prob_spk_dir=prob_spk_dir,
        cid=4,
        wid=10,
        ntp=0,
        _func_label="show_single_spike_resp_410",
        _transparent=True,
    )
    show_single_spike_resp(
        prob_spk_dir=prob_spk_dir,
        cid=4,
        wid=15,
        ntp=0,
        _func_label="show_single_spike_resp_415",
        _transparent=True,
    )
    show_tline_sample(
        kappa_dir=kappa_dir,
        cid=4,
        wid=10,
        err_method="std",
        err_std=1.96,
        _func_label="show_tline_example_410",
        _transparent=True,
    )
    show_tline_sample(
        kappa_dir=kappa_dir,
        cid=4,
        wid=15,
        err_method="std",
        err_std=1.96,
        _func_label="show_tline_example_415",
        _transparent=True,
    )
    draw_entire_tl(
        kappa_dir=kappa_dir, err_std=1.96, p_ranges=(2.5, 97.5), _transparent=True
    )


if __name__ == "__main__":
    main()
