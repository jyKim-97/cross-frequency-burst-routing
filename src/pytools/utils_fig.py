import pickle as pkl
import xarray as xa
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

cm = 1 / 2.54

fdir_cur = "../figures"


landmark_points = [
    dict(a=1.29, b=0.5, w=0.85, id_sync=0),
    dict(a=1, b=0.78, w=-0.3, id_sync=0),
    dict(a=1.28, b=0.71, w=0.9, id_sync=2),
    dict(a=1.14, b=0.29, w=0.15, id_sync=1),
    dict(a=0.85, b=0.5, w=-0.3, id_sync=2),
    dict(a=0.85, b=0.85, w=-0.3, id_sync=2),
    dict(a=0.57, b=0.14, w=0.15, id_sync=2),
]


def save_pickle(fname, data, replace=False):
    if not replace and os.path.isfile(fname):
        raise ValueError("File %s exist" % (fname))

    with open(fname, "wb") as fp:
        pkl.dump(fp)


def load_pickle(fname):
    with open(fname, "rb") as fp:
        return pkl.load(fp)


def load_dataarray(fname):
    if os.path.exists(fname):
        dataarray = xa.open_dataarray(fname)
        return dataarray
    else:
        raise FileNotFoundError(f"File {fname} does not exist.")


def load_dataset(fname):
    if os.path.exists(fname):
        dataset = xa.open_dataset(fname)
        return dataset
    else:
        raise FileNotFoundError(f"File {fname} does not exist.")


def set_plt(fdir_out: None | str = None):
    font_files = fm.findSystemFonts(fontpaths="./font")
    for font_file in font_files:
        fm.fontManager.addfont(font_file)

    plt.rcParams["mathtext.fontset"] = "dejavusans"

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["text.usetex"] = False

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.bottom"] = True
    plt.rcParams["axes.spines.left"] = True

    plt.rcParams["xtick.labelsize"] = 5.5
    plt.rcParams["ytick.labelsize"] = 5.5
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["figure.dpi"] = 100

    plt.rcParams["lines.linewidth"] = 1.2

    if fdir_out is not None:
        import os

        global fdir_cur
        fdir_cur = fdir_out
        os.makedirs(fdir_out, exist_ok=True)


def show_spline(ax, top=False, right=False, bottom=False, left=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_visible(left)


def show_scalebar(
    ax,
    size=10,
    label="1 s",
    vh="horizontal",
    anchor_pos=None,
    pad=0.5,
    lw=2,
    color="k",
    fontsize=12,
):
    """
    ------
    Parameters
    ax: matplotlib.axes.Axes
        The axes to draw the scale bar on.
    size: float
        Length of the scale bar in data coordinates.
    label: str
        Label for the scale bar.
    vh: str
        Orientation of the scale bar. Options are 'horizontal' or 'vertical'.
    anchor_pos: tuple
        Position of the scale bar in data coordinates. If None, the position is
        determined automatically.
    pad: float
        Padding between the scale bar and the label.
    lw: float
        Line width of the scale bar.
    color: str or tuple
        Color of the scale bar.
    fontsize: int
        Font size of the label.
    -------
    """

    assert vh in ["horizontal", "vertical"], "vh must be 'horizontal' or 'vertical'"

    xl = ax.get_xlim()
    yl = ax.get_ylim()
    dx = xl[1] - xl[0]
    dy = yl[1] - yl[0]

    if anchor_pos is None:
        y0 = yl[0] + 0.1 * dy
        x0 = xl[0] + 0.1 * dx
    else:
        assert len(anchor_pos) == 2, "anchor_pos must be a tuple of (x, y)"
        x0, y0 = anchor_pos

    if vh == "horizontal":
        ax.plot([x0, x0 + size], [y0, y0], color=color, lw=lw)
        ax.text(
            x0 + size / 2,
            y0 - pad,
            label,
            ha="center",
            va="top",
            color=color,
            fontsize=fontsize,
        )
    elif vh == "vertical":
        ax.plot([x0, x0], [y0, y0 + size], color=color, lw=lw)
        ax.text(
            x0 - pad,
            y0 + size / 2,
            label,
            ha="right",
            va="center",
            rotation=90,
            color=color,
            fontsize=fontsize,
        )


def get_axlim(ax):
    return ax.get_xlim(), ax.get_ylim()


def get_subax_pos(num_row, num_col, space_row=0.12, space_col=0.1):
    get_w = lambda num, space: (1 - (num + 1) * space) / num

    wr = get_w(num_row, space_row)
    wc = get_w(num_col, space_col)

    pos = []
    for nr in range(num_row):
        pos.append([])
        y0 = space_row + (wr + space_row) * nr
        for nc in range(num_col):
            x0 = space_col + (wc + space_col) * nc
            pos[-1].append([x0, y0, wc, wr])
    return pos


def get_custom_subplots(
    h_ratio,
    w_ratio,
    h_blank_interval_set=None,
    w_blank_interval_set=None,
    h_blank_interval=0.05,
    w_blank_interval=0.05,
    h_blank_boundary=0.05,
    w_blank_boundary=0.05,
):

    nh, nw = len(h_ratio), len(w_ratio)
    if h_blank_interval_set is not None:
        assert len(h_blank_interval_set) == nh - 1
    else:
        h_blank_interval_set = [h_blank_interval] * (nh - 1)
    h_blank_interval_set = list(h_blank_interval_set) + [0]

    if w_blank_interval_set is not None:
        assert len(w_blank_interval_set) == nw - 1
    else:
        w_blank_interval_set = [w_blank_interval] * (nw - 1)
    w_blank_interval_set = list(w_blank_interval_set) + [0]

    assert np.sum(h_blank_interval_set) < 1
    assert np.sum(w_blank_interval_set) < 1

    h_set = (
        np.array(h_ratio)
        / np.sum(h_ratio)
        * (1 - 2 * h_blank_boundary - np.sum(h_blank_interval_set))
    )
    w_set = (
        np.array(w_ratio)
        / np.sum(w_ratio)
        * (1 - 2 * w_blank_boundary - np.sum(w_blank_interval_set))
    )

    ax_set = []
    h0 = 1 - h_blank_boundary
    for nr in range(nh):
        ax_set.append([])
        w0 = w_blank_boundary
        for nc in range(nw):
            if w0 < 0 or h0 - h_set[nr] < 0 or (w0 + w_set[nc]) > 1 or h0 > 1:
                raise ValueError("axis position exceeds the predefined figure")

            ax = plt.axes((w0, h0 - h_set[nr], w_set[nc], h_set[nr]))

            ax_set[-1].append(ax)
            w0 += w_set[nc] + w_blank_interval_set[nc]
        h0 -= h_set[nr] + h_blank_interval_set[nr]

    return ax_set


def remove_ticklabels(ax):
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def save_fig(fname):
    fname = fname.split(".")[0]
    plt.savefig(
        os.path.join(fdir_cur, fname + ".png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.savefig(
        os.path.join(fdir_cur, fname + ".svg"),
        dpi=1200,
        bbox_inches="tight",
        transparent=True,
    )
    print("Saved figure to %s" % (os.path.join(fdir_cur, fname)))


def backup_fig():
    import shutil
    from datetime import datetime

    x = datetime.now()
    fdir_backup = os.path.join(fdir_prev, x.strftime("%y%m%d"))
    if not os.path.exists(fdir_backup):
        os.makedirs(fdir_backup)

    print("Copy files to %s" % (fdir_backup))
    fnames = [f for f in os.listdir() if "figure" in f and "ipynb" in f]
    print(fnames)
    for f in fnames:
        shutil.copyfile(f, os.path.join(fdir_backup, f))
    shutil.copytree("./figures", os.path.join(fdir_backup, "figures"))


def brighten_hex(hex_color, factor=1.2):
    import matplotlib.colors as mcolors

    """
    Brighten a hex color by scaling its RGB values.
    
    Parameters:
        hex_color (str): A hex color string, e.g., '#1f77b4'
        factor (float): Brightness scaling factor (>1 to brighten)
    
    Returns:
        str: Brightened hex color
    """
    rgb = mcolors.to_rgb(hex_color)
    bright_rgb = tuple(min(1, c * factor) for c in rgb)
    return mcolors.to_hex(bright_rgb)


def read_motif(lb):
    assert lb[0] == "F"

    mid = np.zeros(4)
    mid[0] = lb[2] == "f"
    mid[1] = lb[3] == "s"
    mid[2] = lb[7] == "f"
    mid[3] = lb[8] == "s"

    return mid


def draw_motif_pictogram(lb, rcolor="k", c_pict="#7d0000"):
    from matplotlib.patches import Circle, FancyBboxPatch

    mid = read_motif(lb)

    r = 0.8
    x0 = 2
    y0 = 9
    dy1 = 2
    dy2 = 3.0

    ax = plt.gca()

    wbig = 2
    wb = 0.5
    w = 2
    robj_big = FancyBboxPatch(
        (x0 - wbig, y0 - 3 * dy1 - dy2 + wb),
        2 * wbig,
        4 * dy1 + dy2 - 2 * wb,
        facecolor=rcolor,
        edgecolor="none",
        boxstyle="round, pad=0.5",
    )
    robj_top = FancyBboxPatch(
        (x0 - w / 2, y0 - dy1 / 2 * 3),
        w,
        2 * dy1,
        edgecolor=rcolor,
        facecolor="w",
        lw=0.5,
        boxstyle="round, pad=0.3",
    )
    robj_bot = FancyBboxPatch(
        (x0 - w / 2, y0 - dy2 - dy1 / 2 * 5),
        w,
        2 * dy1,
        edgecolor=rcolor,
        facecolor="w",
        lw=0.5,
        boxstyle="round, pad=0.3",
    )
    ax.add_patch(robj_big)
    ax.add_patch(robj_top)
    ax.add_patch(robj_bot)

    y = y0
    for n in range(4):
        if mid[n] == 1:
            cobj = Circle((x0, y), radius=r, facecolor=c_pict)
            ax.add_patch(cobj)

        if n == 1:
            y -= dy2
        else:
            y -= dy1

    plt.xlim([-0.5, 4.5])
    plt.ylim([-2, 12])
    plt.axis("off")
    plt.axis("equal")


def draw_landmark_diagram(
    cid=-1,
    a=1,
    b=1,
    w=1,
    id_sync=1,
    ax=None,
    color_e="#a82a2f",
    colors_i=("#2e3191", "#387a63"),
    edgecolor=(0.4, 0.4, 0.4),
    text_pops=("Fast", "Slow"),
    fontsize=6,
    rot=0,
    box_color=None,
):
    """
    cid: cluster ID (based on landmark_points), alwyas has priority
    """

    from matplotlib import patches
    import matplotlib.transforms as transforms
    from matplotlib.patches import FancyBboxPatch

    if cid != -1:
        assert 0 < cid <= 7, "Cluster ID index mismatches"
        a, b, w, id_sync = landmark_points[cid - 1].values()

    color_sync = ("#6d6e71", "#a7a9ac", "#e6e7e8")

    r = 2
    d = 0.2

    a_max = 1.3
    b_max = 0.8
    w_max = 1
    mul = 1.5

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    xf, xs = 0, 9 / 2 * r
    if box_color is not None:
        wbig_x, wbig_y = 2.6, 2.1
        robj_big = FancyBboxPatch(
            (xf - wbig_x, -wbig_y),
            xs - xf + 2 * wbig_x + 0.4,
            2 * r + 2 * wbig_y,
            edgecolor=box_color,
            linewidth=2,
            facecolor="none",
            boxstyle="round, pad=0.5",
        )
        ax.add_patch(robj_big)

    r1 = patches.Circle(
        (xf, r), r, facecolor=color_sync[id_sync], edgecolor=edgecolor, lw=0.2
    )
    r2 = patches.Circle(
        (xs, r), r, facecolor=color_sync[id_sync], edgecolor=edgecolor, lw=0.2
    )
    ax.add_patch(r1)
    ax.add_patch(r2)

    x1, x2 = xf + r + 0.5, xs - r - 0.5
    wf = 1 - w if w > 0 else 1
    ws = 1 + w if w < 0 else 1

    a_f, b_f = a * wf / a_max * mul, b * wf / b_max * mul
    a_s, b_s = a * ws / a_max * mul, b * ws / b_max * mul

    bmax, bmin = max(b_f, b_s), min(b_f, b_s)
    d_i = (3 / 4 * bmax + bmin / 2 + d) / 2

    xy_f, w_f = [x1, r + d_i - b_f / 2], x2 - x1 - 3 / 4 * b_f
    xy_s, w_s = [x1 + 3 / 4 * b_s, r - d_i - b_s / 2], x2 - x1 - 3 / 4 * b_s
    y_f = xy_f[1] + 5 / 4 * b_f + 3 / 4 * a_f + 0.2
    y_s = xy_s[1] - 1 / 4 * b_s - 3 / 4 * a_s - 0.2

    d2 = y_f - ((y_f - y_s) / 2 + r)
    xy_f[1] -= d2
    xy_s[1] -= d2
    y_f -= d2
    y_s -= d2

    dr_f, dr_s = y_f - r, r - y_s

    ri_f = patches.Rectangle(xy_f, w_f, b_f, color=colors_i[0], lw=0)
    ci_f = patches.Circle(
        (xy_f[0] + w_f, xy_f[1] + b_f / 2), 3 / 4 * b_f, color=colors_i[0], lw=0
    )
    ri_s = patches.Rectangle(xy_s, w_s, b_s, color=colors_i[1], lw=0)
    ci_s = patches.Circle(
        (xy_s[0], xy_s[1] + b_s / 2), 3 / 4 * b_s, color=colors_i[1], lw=0
    )

    re_f = patches.Rectangle((x1, y_f - a_f / 2), x2 - x1, a_f, color=color_e, lw=0)
    te_f = patches.Polygon(
        [(x2, y_f - 3 / 4 * a_f), (x2, y_f + 3 / 4 * a_f), (x2 - r / 2, y_f)],
        color=color_e,
    )
    re_s = patches.Rectangle((x1, y_s - a_s / 2), x2 - x1, a_s, color=color_e, lw=0)
    te_s = patches.Polygon(
        [(x1, y_s - 3 / 4 * a_s), (x1, y_s + 3 / 4 * a_s), (x1 + r / 2, y_s)],
        color=color_e,
    )

    ax.add_patch(ri_f)
    ax.add_patch(ci_f)
    ax.add_patch(ri_s)
    ax.add_patch(ci_s)

    ax.add_patch(re_f)
    ax.add_patch(te_f)
    ax.add_patch(re_s)
    ax.add_patch(te_s)

    plt.xlim([xf - r - 0.2, xs + r + 0.2])
    plt.ylim([xf - r - 0.2 - 2, xs + r + 0.2 - 2])
    plt.xticks(np.linspace(xf - r - 0.2, xs + r + 0.2, 5))
    plt.axis("equal")
    plt.axis("off")

    t = transforms.Affine2D().rotate_deg_around((xf + xs) / 2, r, -rot) + ax.transData

    for p in ax.patches:
        p.set_transform(t)

    text_opt = dict(ha="center", va="center", fontsize=fontsize, transform=t)
    plt.text(xf, r, text_pops[0], **text_opt)
    plt.text(xs, r, text_pops[1], **text_opt)


def get_cid_color(cid, cid_max=7, cmap="turbo"):
    assert 0 < cid <= cid_max
    palette = plt.get_cmap(cmap)
    return palette((cid - 1) / (cid_max - 1))


def set_phase_ticks(ax=None, xy="x", div=2):
    """
    Set ticks for phase axes (0 to 2π).

    Parameters:
        ax: matplotlib.axes.Axes, optional
            The axes to set the ticks on. If None, uses the current axes.
            The axes to set the ticks on. If None, uses the current axes.
        xy: str
            Axis to set ticks on ('x' or 'y').
        div (int)
    """
    if ax is None:
        ax = plt.gca()
    assert xy in ("x", "y"), "xy must be 'x' or 'y'"

    ticks_int = np.arange(-div, div + 0.1)
    tick_labels = []
    for tt in ticks_int:
        if tt == -div:
            tick_labels.append(r"$-\pi$")
        elif tt == div:
            tick_labels.append(r"$\pi$")
        elif tt == 0:
            tick_labels.append(r"$0$")
        else:
            tick_labels.append(r"$%d/%d\pi$" % (tt, div))
    ticks = ticks_int / div * np.pi

    if xy == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
    elif xy == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)


def generate_custom_subplots(
    ratio_row, ratio_col, space_row=0.05, space_col=0.05, empty_row=0.1, empty_col=0.1
):
    """
    Generates custom subplot axes based on row and column ratios.

    Parameters:
    - ratio_row: List[float], ratios for each row (height proportions)
    - ratio_col: List[float], ratios for each column (width proportions)
    - space_row: float, spacing between rows (0 < space < 1)
    - space_col: float, spacing between columns (0 < space < 1)
    - empty_row: float, margin on top and bottom
    - empty_col: float, margin on left and right

    Returns:
    - fig: matplotlib.figure.Figure
    - axes: 2D list of AxesSubplot objects (rows x columns)
    """

    total_height = sum(ratio_row)
    total_width = sum(ratio_col)
    nrows = len(ratio_row)
    ncols = len(ratio_col)

    total_vspace = empty_row * 2 + space_row * (nrows - 1)
    total_hspace = empty_col * 2 + space_col * (ncols - 1)

    if total_vspace >= 1.0 or total_hspace >= 1.0:
        raise ValueError("Margins and spacings are too large. Reduce them.")

    height_unit = (1.0 - total_vspace) / total_height
    width_unit = (1.0 - total_hspace) / total_width

    row_heights = [r * height_unit for r in ratio_row]
    row_bottoms = []
    current_bottom = 1.0 - empty_row
    for h in row_heights:
        current_bottom -= h
        row_bottoms.append(current_bottom)
        current_bottom -= space_row

    col_widths = [c * width_unit for c in ratio_col]
    col_lefts = []
    current_left = empty_col
    for w in col_widths:
        col_lefts.append(current_left)
        current_left += w + space_col

    fig = plt.figure()
    axes = []
    for i, (bottom, height) in enumerate(zip(row_bottoms, row_heights)):
        row_axes = []
        for j, (left, width) in enumerate(zip(col_lefts, col_widths)):
            ax = fig.add_axes([left, bottom, width, height])
            row_axes.append(ax)
        axes.append(row_axes)

    return fig, axes


def get_figure(figsize=(1, 1), dpi=200):

    fig = plt.figure(figsize=(figsize[0] * cm, figsize[1] * cm), dpi=dpi)
    return fig
