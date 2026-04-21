import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from pytools import hhsignal, hhtools
from pytools import utils_fig as uf

uf.set_plt()

import figure_manager as fm

fsample = "../results/singlepop/activity_sample.pkl"
fsample_power = "../results/singlepop/sample_detail.pkl"
fburst_power = "../results/singlepop/burst_prop2.pkl"
fpost_fast = "../results/singlepop/pe_nu_mfast.nc"
fpost_slow = "../results/singlepop/pe_nu_mslow.nc"
craster = ("#d70606", "#003f5c")


def compute_mua(detail, st=0.001, srate=1000, dt=0.01):

    t0, t1 = detail["ts"][0], detail["ts"][-1]
    t = np.arange(t0, t1 + 1 / 2 / srate, 1 / srate)
    mua = np.zeros_like(t)
    for step_spk in detail["step_spk"]:
        for n in step_spk:
            nt = int((n * dt * 1e-3 - t0) * srate)
            mua[nt] += 1

    s = st * srate
    mua = gaussian_filter1d(mua, s)
    return mua, t


def _remove_axis():
    plt.xticks([])
    plt.yticks([])
    uf.show_spline(plt.gca())


def rotate(r, agl):
    R = np.array([[np.cos(agl), np.sin(agl)], [-np.sin(agl), np.cos(agl)]])
    return R @ r


def draw_rot_line(x0, y0, s0, theta, L=1, lw=0.8, align="center", **kwargs):
    ax = plt.gca()
    to_pix = ax.transData.transform
    to_data = ax.transData.inverted().transform

    dv_data = np.array([1.0, s0], dtype=float)
    dv_data /= np.linalg.norm(dv_data)

    p0 = np.array(to_pix([x0, y0]))
    pT = np.array(to_pix([x0 + dv_data[0], y0 + dv_data[1]]))
    t_pix = pT - p0

    n_pix = rotate(t_pix, theta)
    n_pix /= np.linalg.norm(n_pix)

    if align == "center":
        half = L / 2.0
        p1 = p0 - half * n_pix
        p2 = p0 + half * n_pix
    elif align == "right":
        p1 = p0
        p2 = p0 + L * n_pix
    elif align == "left":
        p1 = p0 - L * n_pix
        p2 = p0
    else:
        raise ValueError("")

    x1, y1 = to_data(p1)
    x2, y2 = to_data(p2)

    plt.plot((x1, x2), (y1, y2), "k", lw=lw, **kwargs)


def draw_curved_ticks(x, y, xticks, l=1.0, **kwargs):
    from scipy.interpolate import interp1d

    slopes = np.gradient(y, x)

    xq = xticks
    yq = interp1d(x, y)(xq)
    sq = interp1d(x, slopes)(xq)

    for xi, yi, si in zip(xq, yq, xq):

        draw_rot_line(xi, yi, si, 120 / 180 * np.pi, L=4, lw=0.8)

    draw_rot_line(
        x[-1], y[-1], slopes[-1], 150 / 180 * np.pi, L=5, lw=0.8, align="right"
    )
    draw_rot_line(
        x[-1], y[-1], slopes[-1], 210 / 180 * np.pi, L=5, lw=0.8, align="right"
    )
    plt.plot(x, y, "k-", lw=0.8)


def draw_echelon(y, pl, d, a=1):

    pl = (pl[0] * a, pl[1] * a)
    d = d / np.sqrt(a)

    dp = (pl[1] - pl[0]) / 10
    ysub = np.linspace(pl[0], pl[1] + dp, 31)
    xsub = d * np.sqrt(ysub) / 1000

    yq = np.linspace(pl[0], pl[1], 3)
    xq = d * np.sqrt(yq) / 1000

    draw_curved_ticks(xsub, ysub, xq, l=2)

    nq = (0, 0.5, 1)
    dx = (xq[1] - xq[0]) / 5
    for x0, y0, n0 in zip(xq, yq, nq):
        plt.text(
            x0 + dx, y0 - dp * 1.2, "%.1f" % (n0), fontsize=5, ha="center", va="center"
        )


@fm.figure_renderer(fig_name="sample")
def draw_sample_activity(figsize=(14, 2.5), pop_id=0, xl=(1500, 2000), fdata=None):
    if fdata is None:
        fdata = fsample

    fig = uf.get_figure(figsize)

    data = uf.load_pickle(fdata)
    details = data["details"][pop_id]

    plt.axes(position=(0.1, 0.55, 0.8, 0.4))
    hhtools.draw_spk(
        details["step_spk"], xl=xl, colors=craster, color_ranges=(800, 1000), ms=0.8
    )
    plt.ylim([500, 900])
    _remove_axis()

    plt.axes(position=(0.1, 0.35, 0.8, 0.2))
    mua, t = compute_mua(details, st=0.001, srate=1000)
    t *= 1e3

    idt = (t >= xl[0]) & (t < xl[1])
    plt.fill_between(
        t[idt],
        np.zeros_like(mua[idt]) - 1,
        mua[idt],
        color=(0.9, 0.9, 0.9),
        edgecolor="k",
        linewidth=1,
    )
    plt.xlim(xl)
    plt.ylim([0, 60])
    _remove_axis()

    plt.axes(position=(0.1, 0.05, 0.8, 0.22))
    t = details["ts"] * 1e3
    idt = (t >= xl[0]) & (t < xl[1])
    plt.plot(t[idt], details["vlfp"][idt], c="k", lw=1)
    plt.xlim(xl)
    plt.ylim([-8, 8])
    _remove_axis()

    return fig


@fm.figure_renderer(fig_name="scalebar_h")
def draw_scalebar_h(figsize=(14, 8)):
    fig = uf.get_figure(figsize)
    uf.show_scalebar(
        plt.gca(),
        size=100,
        label="100 ms",
        anchor_pos=(1100, 400),
        color="k",
        lw=2,
        pad=3,
        fontsize=10,
    )
    plt.xlim((1000, 2000))
    plt.ylim([380, 410])
    plt.axis("off")
    return fig


@fm.figure_renderer(fig_name="scalebar_v")
def draw_scalebar_v(figsize=(14, 0.6)):
    fig = uf.get_figure(figsize)
    uf.show_scalebar(
        plt.gca(),
        size=20,
        label="20 %",
        anchor_pos=(10, 10),
        color="k",
        lw=2,
        pad=3,
        vh="vertical",
        fontsize=10,
    )
    plt.xlim([0, 500])
    plt.ylim([0, 60])
    plt.axis("off")
    return fig


@fm.figure_renderer(fig_name="scalebar_vlfp")
def draw_scalebar_vlfp(figsize=(14, 0.4)):
    fig = uf.get_figure(figsize)
    uf.show_scalebar(
        plt.gca(),
        size=10,
        label="10 mV",
        anchor_pos=(10, -5),
        color="k",
        lw=2,
        pad=3,
        vh="vertical",
        fontsize=10,
    )
    plt.xlim([0, 500])
    plt.ylim([-7, 7])

    plt.axis("off")
    return fig


@fm.figure_renderer(fig_name="echelon_space")
def draw_parameter_space(figsize=(3.5, 9.2), fpost="", a=0, d=0.0, pl=[0, 1]):

    def show_contourf(im_array, vmin=0.0, vmax=1.0, num=21, cmap="turbo", dc=0.1):
        x = im_array.nu / 1000
        y = im_array.pe * a
        im = gaussian_filter(im_array.data, 1)

        plt.contourf(
            x,
            y,
            im,
            np.concatenate((np.linspace(vmin, vmax, num), [200])),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        cbar = plt.colorbar(ticks=np.arange(vmin, vmax + dc / 2, dc))
        cbar.ax.set_ylim([vmin, vmax])
        cbar.ax.tick_params(length=3)

    def _set_label():
        plt.xlabel(r"$\nu$ (kHz)", labelpad=1)
        plt.ylabel(r"$p_I$", labelpad=2)

        plt.xticks(xticks)
        plt.xlim([xmin, xmax])
        plt.yticks(yticks)

    da = uf.load_dataarray(fpost)
    da = da.sel(dict(type="mean"))

    y = da.pe.values

    if "fast" in fpost:

        xmin, xmax, xticks, yticks = 2, 8, (2, 4, 6, 8), (0.2, 0.4, 0.6, 0.8)
    elif "slow" in fpost:

        xmin, xmax, xticks, yticks = 2, 6, (2, 4, 6), (0.05, 0.15, 0.25, 0.35)
    else:
        raise ValueError("%s is not identified" % (fpost))

    fig = uf.get_figure(figsize)
    pos = uf.get_subax_pos(3, 1, space_row=0.12, space_col=0.1)

    plt.axes(pos[-1][0])
    im_array = da.sel(dict(vars="chi"))
    show_contourf(im_array, vmin=0, vmax=0.6, dc=0.2)
    plt.title(r"Synchrony level, $\chi$", pad=4)
    draw_echelon(y, pl, d, a=a)
    _set_label()

    plt.axes(pos[-2][0])
    im_array = da.sel(dict(vars="fr"))
    show_contourf(im_array, vmin=0, vmax=20, dc=5)
    plt.title(r"Firing rate, $z$", pad=4)
    draw_echelon(y, pl, d, a=a)
    _set_label()

    plt.axes(pos[-3][0])
    im_array = da.sel(dict(vars="fnet"))
    show_contourf(im_array, vmin=10, vmax=90, dc=20)
    plt.title(r"Network frequency, $f_{net}$", pad=4)
    draw_echelon(y, pl, d, a=a)
    _set_label()

    return fig


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


def _count(matsize, idx, idy):
    num = np.zeros(matsize)
    for n in range(len(idx)):
        num[idy[n], idx[n]] += 1
    return num


def get_mid_pts(edges):
    return (edges[1:] + edges[:-1]) / 2


@fm.figure_renderer(fig_name="burstmap")
def show_burstmap(figsize=(7.8, 3.3), pop_id=0, fdata=None, fburst=None):
    if fdata is None:
        fdata = fsample_power

    if fburst is None:
        fburst = fburst_power

    sdata = uf.load_pickle(fdata)
    detail = sdata["detail"]
    psd, fpsd, tpsd = hhsignal.get_stfft(
        detail["vlfp"][pop_id + 1], detail["ts"], 2000, frange=(10, 90)
    )
    bdata = uf.load_pickle(fburst)
    mbin_t = bdata["attrs"]["mbin_t"]

    bf = np.array(bdata["burst_props"][pop_id]["burst_f"])
    bl = np.array(bdata["burst_props"][pop_id]["burst_len"])
    ba = np.array(bdata["burst_props"][pop_id]["burst_amp"])

    idx = bl * mbin_t >= 3 / bf
    bf = bf[idx]
    ba = ba[idx]
    bl = bl[idx] * mbin_t

    if pop_id == 0:
        xedges = np.linspace(-10, 120, 21) * mbin_t
        yedges = np.linspace(35, 105, 21)
        xticks = np.arange(0, 1, 0.2)
        yticks = np.arange(40, 101, 10)
        yl = (38, 92)
        xl = (0, 0.8)
        lb = "Fast"
    else:
        xedges = np.linspace(-10, 120, 21) * mbin_t
        yedges = np.linspace(20, 50, 21)
        xticks = np.arange(0.0, 1.05, 0.2)
        yticks = np.arange(20, 51, 5)
        yl = (25, 45)
        xl = (0, 1)
        lb = "Slow"

    fig = uf.get_figure(figsize)

    im = hist2d(bf, bl, yedges, xedges)
    im = gaussian_filter(im, 0.8)
    x = get_mid_pts(xedges)
    y = get_mid_pts(yedges)

    fig = uf.get_figure(figsize)

    plt.axes(position=(0.05, 0.1, 0.38, 0.8))
    plt.imshow(
        psd,
        aspect="auto",
        cmap="jet",
        origin="lower",
        vmin=0.1,
        vmax=0.8,
        extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
        interpolation="bicubic",
    )

    plt.xticks(np.arange(1, 5))
    plt.yticks(np.arange(10, 91, 10))
    plt.colorbar()
    plt.ylim([20, 80])

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.axes(position=(0.57, 0.1, 0.38, 0.8))
    plt.contourf(
        x,
        y,
        im / im.sum(),
        np.concatenate((np.linspace(0, 0.04, 21), [1])),
        cmap="turbo",
        vmax=0.04,
        vmin=0,
    )
    cbar = plt.colorbar(ticks=[0, 0.01, 0.02, 0.03, 0.04])
    cbar.ax.set_ylim([0, 0.04])

    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim(xl)

    plt.ylim(yl)

    plt.xlabel("Burst duration (s)")
    plt.ylabel("Frequency (Hz)")

    return fig


if __name__ == "__main__":
    draw_sample_activity(xl=(1500, 2000), pop_id=0, _func_label="sample_fast")
    draw_sample_activity(xl=(1800, 2300), pop_id=1, _func_label="sample_slow")

    draw_parameter_space(
        fpost=fpost_fast,
        d=14142.14,
        pl=[0.051, 0.234],
        a=3,
        _func_label="echelon_fast",
        _transparent=True,
    )
    draw_parameter_space(
        fpost=fpost_slow,
        d=15450.58,
        pl=[0.028, 0.105],
        a=2.5,
        _func_label="echelon_slow",
        _transparent=True,
    )
    show_burstmap(pop_id=0, _func_label="bmap_fast")
    show_burstmap(pop_id=1, _func_label="bmap_slow")
