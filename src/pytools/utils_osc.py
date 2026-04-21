import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Pool
import pickle as pkl
from typing import Tuple

from functools import partial

import os
import sys

sys.path.append("/home/jungyoung/Project/hh_neuralnet/include/")
try:
    from . import hhsignal, hhtools
except ImportError:
    import hhsignal
    import hhtools


prefix_data = "/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/data"
prefix_motif = "/home/jungyoung/Project/hh_neuralnet/extract_osc_motif/data/osc_motif"


cw_pair = [
    [1, 0],
    [1, 2],
    [1, 4],
    [1, 6],
    [2, 0],
    [2, 2],
    [2, 8],
    [2, 10],
    [3, 0],
    [3, 2],
    [3, 4],
    [3, 6],
    [4, 0],
    [4, 2],
    [4, 5],
    [4, 7],
    [5, 0],
    [5, 4],
    [5, 10],
    [5, 14],
    [6, 0],
    [6, 4],
    [6, 10],
    [6, 14],
    [7, 0],
    [7, 10],
    [8, 0],
    [8, 2],
    [8, 13],
    [8, 15],
]


def set_default_dir(fdir):
    global default_dir
    default_dir = fdir


def par_func(f, arg_set, num_process, desc="", verbose=True):
    N = len(arg_set)
    result_set = []
    if num_process == 1:
        for arg in tqdm(arg_set, desc=desc):
            result_set.append(f(arg))
    else:
        pool = Pool(processes=num_process)
        if verbose:
            for result in tqdm(pool.imap(f, iterable=arg_set), total=N, desc=desc):
                result_set.append(result)
        else:
            for result in pool.imap(f, iterable=arg_set):
                result_set.append(result)

    return result_set


def load_osc_motif(cid, wid, tag="", reverse=False, verbose=False):

    fname = prefix_motif + tag
    fname = os.path.join(fname, "motif_info_%d" % (cid))
    if reverse:
        fname = fname + "(low)"

    with open(fname + ".pkl", "rb") as fp:
        osc_motif = pkl.load(fp)

    update_date = osc_motif["metainfo"]["last-updated"]

    winfo = osc_motif["winfo"][wid]
    if len(winfo) == 0:
        print("Word ID %2d does not exist in cluster%d" % (wid, cid))
    elif verbose:
        print("Loaded oscillation motif information udpated in %s" % (update_date))
        print("%4d motifs are detected" % (len(winfo)), end=",")

    return osc_motif["winfo"][wid], update_date


def collect_chunk(
    cid: int,
    wid: int,
    summary_obj=None,
    tag="",
    target="lfp",
    st_mua=1e-3,
    dt=0.01,
    nequal_len: int = None,
    nadd: int = 0,
    teq: Tuple = (0.5, -0.5),
    norm=False,
    filt_range: Tuple = None,
    srate=2000,
    reverse=False,
    verbose=True,
):
    """
    Collect data chunk

    nequal_len: stack v_set with the same length
    reverse: boolean
        add more points on the right (True)

    """

    if summary_obj is None:
        print("load default dataset in %s" % (default_dir))
        summary_obj = hhtools.SummaryLoader(default_dir, load_only_control=True)

    winfo, _ = load_osc_motif(cid, wid, tag=tag, reverse=False, verbose=verbose)

    assert target in ("lfp", "mua")
    if target == "lfp":
        _read_value = lambda detail_data: detail_data["vlfp"][1:]
    else:
        _read_value = lambda detail_data: detail_data["mua"]

    if filt_range is not None:
        pre_sos = hhsignal.get_sosfilter(filt_range, srate)

        def _get_value(detail_data):
            x = _read_value(detail_data)
            return hhsignal.filt(_read_value(x[0]), pre_sos), hhsignal.filt(
                _read_value(x[1]), pre_sos
            )

    else:
        _get_value = lambda detail_data: _read_value(detail_data)

    def _norm(x):
        if norm:
            return (x - x.mean()) / x.std()
        else:
            return x

    if nequal_len is not None:
        nequal_len += nadd

    chunk = []
    nitr_prv = -1

    if verbose:
        _range = partial(
            trange, desc="collect %s in %d%02d" % (target, cid, wid), ncols=100
        )
    else:
        _range = range

    for i in _range(len(winfo)):
        nitr = winfo[i][0]
        tl = winfo[i][1]

        if nitr != nitr_prv:
            detail_data = summary_obj.load_detail(cid - 1, nitr)
            x1, x2 = _get_value(detail_data)
            nitr_prv = nitr

        if (tl[0] < teq[0]) or (tl[1] > detail_data["ts"][-1] + teq[1]):
            continue

        nr = ((tl - detail_data["ts"][0]) * srate).astype(int)

        if not reverse:
            nr[0] -= nadd
        else:
            nr[1] += nadd

        if nr[0] < teq[0] * srate:
            continue
        if nr[1] > (detail_data["ts"][-1] + teq[1]) * srate:
            continue

        if nequal_len is not None:
            x_sub = np.zeros((2, nequal_len)) * np.nan
            nmax = min(nequal_len, nr[1] - nr[0])
            x_sub[:, :nmax] = np.array(
                [_norm(x1[nr[0] : nr[0] + nmax]), _norm(x2[nr[0] : nr[0] + nmax])]
            )

        else:
            x_sub = [_norm(x1[nr[0] : nr[1]]), _norm(x2[nr[0] : nr[1]])]

        chunk.append(x_sub)

    if nequal_len is not None:
        chunk = np.array(chunk)

    return chunk


def compute_stfft_all(
    vs, ts, frange=(5, 100), mbin_t=0.01, wbin_t=0.5, srate=2000, t0=0.5
):
    psd_set = []
    for v in vs:
        _v, _t = hhsignal.get_eq_dynamics(v, ts, t0)
        psd, fpsd, tpsd = hhsignal.get_stfft(
            _v, _t, srate, mbin_t=mbin_t, wbin_t=wbin_t, frange=frange
        )
        psd_set.append(psd)
    return np.array(psd_set), fpsd, tpsd


def load_pickle(fname):
    """load pickle dataset"""
    with open(fname, "rb") as fp:
        return pkl.load(fp)
