import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from scipy.signal import find_peaks

te_dir = "../results/te/"
mfop_dir = "../results/mfop_results/mfop/"
te_colors = ("#d92a27", "#045894", "#a9a9a9")
c_rect = "#676767"


def load_pickle(fname):
    with open(fname, "rb") as fp:
        return pkl.load(fp)


def write_pickle(fname, data):
    with open(fname, "wb") as fp:
        pkl.dump(data, fp)


def load_te(cid, wid, fdir=te_dir):
    fname = os.path.join(fdir, "te_%d%02d.pkl" % (cid, wid))
    with open(fname, "rb") as fp:
        te = pkl.load(fp)
    return te


def get_max_period(cid, fname=mfop_dir + "amp_range_set.pkl"):
    amp_range = load_pickle(fname)["amp_range_set"][cid]

    fs = [amp_range[k][0] for k in ("fpop", "spop")]
    max_period, nstack = 0, 0
    for f in fs:
        if len(f) > 0:
            max_period += np.average(f)
            nstack += 1
    if nstack == 0:
        max_period = 30
    else:
        max_period /= nstack

    return 1e3 / max_period


def find_te_peaks(te_avg, **kwargs_peak):
    """
    Find peaks in the average TE data.

    Parameters:
    te_avg (array): Average TE values (1D array)

    Returns:
    peaks (array): Indices of the peaks in the TE data.
    """
    peaks, _ = find_peaks(te_avg, **kwargs_peak)
    if te_avg[-3] < te_avg[-2] < te_avg[-1]:
        peaks = np.append(peaks, len(te_avg) - 1)
    if te_avg[0] > te_avg[1] > te_avg[2]:
        peaks = np.insert(peaks, 0, 0)

    return peaks


def _search_overlap(y1, y0, n0, nstep=1):
    """
    Search the overlap when y1 == y0 (assert y1 > y0)
    y1: TE
    y0: threshold
    n0: initial index to start searching
    nstep: step size for searching
    """

    if y1[n0] - y0[n0] < 0:
        raise ValueError("dy should be greater than 0 at n0")

    n = n0
    flag_trough = False
    trough_stack = 0
    nlast_trough = -1
    while y1[n] - y0[n] > 0:
        n += nstep

        if n < 0 or n >= len(y1):
            break
        if y1[n] - y1[n - nstep] > 0:
            if trough_stack == 0:
                nlast_trough = n - nstep
            trough_stack += 1
        else:
            trough_stack = 0

        if trough_stack == 2:
            flag_trough = True
            break

    if flag_trough:
        n = nlast_trough
    n = np.clip(n, 0, len(y1) - 1)
    return n


def identify_inc_points(y1, y0, num_min=4, **kwargs_peak):
    """
    Identify the indices where the y1 > y0
    """

    peaks = find_te_peaks(y1, **kwargs_peak)
    id_inc = []
    for idp in peaks:
        if y1[idp] > y0[idp]:

            idl = _search_overlap(y1, y0, idp, nstep=-1)
            idr = _search_overlap(y1, y0, idp, nstep=1)

            if y0[idl] < y0[idr]:
                thr = y0[idr]
            else:
                thr = y0[idl]
            id_up = np.where(y1[idl : idr + 1] >= thr)[0] + idl
            if len(id_up) < num_min:
                continue

            id_inc.append(list(id_up))

    for id1, id2 in zip(id_inc[:-1], id_inc[1:]):
        if abs(id1[-1] - id2[0]) <= 1:
            id1.pop(-1)
            id2.pop(0)

    return id_inc


def identify_sig_te1d(te1d, prt=95, show_result=False, **kwargs_peak):
    """
    Identify significant TE values in 1D TE data.

    Parameters:
    te1d (dict): Dictionary containing 'te' and 'te_surr' arrays.
    prt (int): Percentile threshold for significance.

    Returns:

    """
    te_true = np.median(te1d["te"], axis=0)
    te_surr = np.percentile(te1d["te_surr"], prt, axis=0)

    id_sig_set = []
    for ndir in range(2):
        y1 = te_true[ndir]
        y0 = te_surr[ndir]

        id_sig_set.append(identify_inc_points(y1, y0, num_min=4, **kwargs_peak))

    if show_result:
        tlag = te1d["tlag"]
        plt.plot(tlag, te_true[0], label="TE True (0)", color=te_colors[0])
        plt.plot(
            tlag,
            te_surr[0],
            label="TE surr thr (0)",
            color=te_colors[0],
            linestyle="--",
        )
        plt.plot(tlag, te_true[1], label="TE True (1)", color=te_colors[1])
        plt.plot(
            tlag,
            te_surr[1],
            label="TE surr thr (1)",
            color=te_colors[1],
            linestyle="--",
        )
        plt.legend()
        plt.xlabel(r"$\tau$ (ms)", fontsize=12)
        plt.ylabel("TE (bits)", fontsize=12)

        for ndir in range(2):
            for id_up in id_sig_set[ndir]:
                plt.plot(
                    tlag[id_up],
                    te_true[ndir][id_up],
                    "k.",
                    label=f"Significant TE (dir={ndir})",
                )

    return id_sig_set


def get_err_range(data, method="quantile", p_ranges=(5, 95), smul=1.96):

    if method == "quantile":
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


def identify_sig_tline(
    kappa_set, err_method="std", err_std=1.96, p_ranges=(5, 95), num_min=2, dtq=1
):

    id_sig_pos = []
    id_sig_neg = []

    t = kappa_set.ndelay.data
    tq = np.arange(t[0], t[-1] + 0.1, dtq)
    for npop in range(2):
        _, ymin_b, ymax_b = get_err_range(
            kappa_set.kappa_base.isel(dict(ntp=1 - npop)).data,
            method=err_method,
            smul=err_std,
            p_ranges=p_ranges,
        )
        ym, ymin, ymax = get_err_range(
            kappa_set.kappa.isel(dict(ntp=1 - npop)).data,
            method=err_method,
            smul=err_std,
            p_ranges=p_ranges,
        )

        ymin_b = np.interp(tq, t, ymin_b)
        ymax_b = np.interp(tq, t, ymax_b)
        ymin = np.interp(tq, t, ymin)
        ymax = np.interp(tq, t, ymax)
        ym = np.interp(tq, t, ym)

        id_sig_pos.append(identify_inc_points(ymin, ymax_b, num_min=num_min))
        id_sig_neg.append(identify_inc_points(-ymax, -ymin_b, num_min=num_min))

    return id_sig_pos, id_sig_neg, tq


def convert_sig_boundary(id_sig_set, t=None):
    tsig_set = []
    for id_sig in id_sig_set:
        tsig_set.append([])
        for ids in id_sig:
            if t is None:
                tsig_set[-1].append([ids[0], ids[-1]])
            else:
                tsig_set[-1].append([t[ids[0]], t[ids[-1]]])
    return tsig_set


def convert_sig_id2time(id_sig_set, tlag):
    """
    Convert significant TE indices to time values.

    Parameters:
    id_sig_set (list): List of significant TE indices for each direction.
    tlag (array): Time lag array corresponding to the TE data.

    Returns:
    list: List of time values for significant TE indices.
    """
    sig_times = []
    for id_sig in id_sig_set:
        sig_times.append([])
        for id_up in id_sig:
            if len(id_up) == 0:
                continue
            sig_times[-1].append([tlag[id_up[0]], tlag[id_up[-1]]])
    return sig_times


def reduce_te_2d(te_data_2d, tcut=None):
    from copy import deepcopy

    if tcut is None:
        tcut = te_data_2d["tlag"][-1]
    assert tcut > te_data_2d["tlag"][0]

    tlag = te_data_2d["tlag"]
    te_data = deepcopy(te_data_2d)

    N = int((tcut - tlag[0]) / (tlag[1] - tlag[0])) + 1
    N = min([N, len(tlag)])

    te_data["tlag"] = te_data["tlag"][:N]
    te_data["te"] = np.zeros((te_data["info"]["ntrue"], 2, N))
    te_data["te_surr"] = np.zeros((te_data["info"]["nsurr"], 2, N))

    for ntp in range(2):
        te_data["te"][:, ntp] = te_data_2d["te"][:, ntp, :N, :N].mean(axis=2)
        te_data["te_surr"][:, ntp] = te_data_2d["te_surr"][:, ntp, :N, :N].mean(axis=2)

    if "info" in te_data.keys():
        te_data["info"]["nmax_delay"] = N

    return te_data


def replace_te_surr2base(te_data, te_base):
    from copy import deepcopy

    te = deepcopy(te_data)

    te["te_surr"] = deepcopy(te_base["te"])
    te["info"]["nsurr"] = te_base["te"].shape[0]

    return te
