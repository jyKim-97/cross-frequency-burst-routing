import numpy as np
import sys

sys.path.append("../include")
try:
    from . import hhsignal
except ImportError:
    import hhsignal

import matplotlib.pyplot as plt


def get_spec_subset(detail, mbin_t=0.05, wbin_t=0.5, srate=2000, frange=(5, 200)):
    psd_set = [None, None]
    psd_set[0], fpsd, tpsd = hhsignal.get_stfft(
        detail["vlfp"][1],
        detail["ts"],
        srate,
        mbin_t=mbin_t,
        wbin_t=wbin_t,
        frange=frange,
    )
    psd_set[1], fpsd, tpsd = hhsignal.get_stfft(
        detail["vlfp"][2],
        detail["ts"],
        srate,
        mbin_t=mbin_t,
        wbin_t=wbin_t,
        frange=frange,
    )
    psd_set = np.array(psd_set)

    idt = tpsd >= 0.5
    psd_set = psd_set[:, :, idt]
    tpsd = tpsd[idt]

    return psd_set, fpsd, tpsd


def get_spec_line(detail, amp_range, params_spec=None):

    if params_spec is None:
        params_spec = dict(mbin_t=0.05, wbin_t=0.5, srate=2000, frange=(5, 200))

    psd_set, fpsd, tpsd = get_spec_subset(detail, **params_spec)

    psd_line = np.zeros((4, len(tpsd)))
    for tp, k in enumerate(("fpop", "spop")):
        for i in range(2):
            if len(amp_range[k][i]) == 0:
                continue

            n = 2 * tp + i
            idf = (fpsd >= amp_range[k][i][0]) & (fpsd < amp_range[k][i][1])
            psd_line[n, :] = psd_set[tp, idf, :].mean(axis=0)

    psd_dict = dict(psd=psd_set, tpsd=tpsd, fpsd=fpsd)

    return psd_line, tpsd, psd_dict


def norm_minmax(arr):
    amax = arr.max(axis=1, keepdims=True)
    amax[amax == 0] = 1
    amin = arr.min(axis=1, keepdims=True)

    return (arr - amin) / (amax - amin)


def digitize(arr, nlevel=10):
    if np.all(arr == 0):
        return arr

    e = np.linspace(0, 1, nlevel + 1)
    e[-1] += 0.05
    ad = np.digitize(arr, e) - 1
    assert (np.min(ad) == 0) and (np.max(ad) == nlevel - 1)
    return ad


def identify_long_seg(arr, min_len=100):
    arr = np.asarray(arr)
    id_seg = []

    start = 0
    while start < len(arr):
        value = arr[start]
        end = start + 1
        while end < len(arr) and arr[end] == value:
            end += 1

        if (end - start) > min_len:
            id_seg.append((start, end, arr[start]))

        start = end

    return id_seg


def show_psd_subset(
    psd_set,
    fpsd,
    tpsd,
    ax_set=None,
    xl=None,
    yl=None,
    interpolation="bicubic",
    vmin=None,
    vmax=None,
    cmap="jet",
):

    draw_new_ax = True if ax_set is None else False
    ax_new = []

    for n in range(2):
        if draw_new_ax:
            ax = plt.subplot(2, 1, n + 1)
        else:
            ax = plt.sca(ax_set[n])
        plt.imshow(
            psd_set[n],
            extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
            cmap=cmap,
            aspect="auto",
            origin="lower",
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlim(xl)
        plt.ylim(yl)
        ax_new.append(ax)

    return ax_new
