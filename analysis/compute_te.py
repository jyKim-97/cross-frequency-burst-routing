"""
This version carefully handles surrogate generation.
Previous versions sampled true and surrogate data separately, which makes difficult to conduct statistical tests on the results.
Instead, this version generates surrogate data based on the true data, ensuring that the statistical properties of the surrogates are consistent with the true data.

- Updated from computeTE4.py
"""

import numpy as np
import argparse
import pickle as pkl
from functools import partial

import os
from pytools import hhtools
from pytools import utils_osc as uo
from pytools import tetools as tt

tag = ""


num_process = 4
srate = 2000

fdir_summary = "../results/twopop_regime_samples/data" + tag

chunk_size = 100
nchunks = 400


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", help="cluster id", required=True, type=int)
    parser.add_argument("--wid", help="word id", required=True, type=int)
    parser.add_argument("--ntrue", default=10, type=int)
    parser.add_argument("--nsurr", default=200, type=int)
    parser.add_argument(
        "--method",
        default="naive",
        type=str,
        choices=("naive", "spo", "mit", "2d", "full", "embedding"),
    )
    parser.add_argument("--target", default="lfp", type=str, choices=("lfp", "mua"))
    parser.add_argument("--nhist", default=1, type=int)
    parser.add_argument("--num_emb_dim", default=2, type=int)
    parser.add_argument("--tlag_max", default=40, type=float)
    parser.add_argument("--tlag_min", default=1, type=float)
    parser.add_argument("--tlag_step", default=0.5, type=float)
    parser.add_argument("--fout", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="Path to save te results. If it is not None, fout will be ignored.",
    )
    parser.add_argument("--tag", default="", choices=("", "_mfast", "_mslow"))
    return parser


def main(
    cid=5,
    wid=10,
    ntrue=10,
    nsurr=1000,
    nhist=1,
    tlag_max=40,
    tlag_min=1,
    tlag_step=1,
    method="naive",
    target="lfp",
    fout=None,
    seed=42,
    verbose=True,
    num_emb_dim=2,
    save_dir=None,
    tag="",
):

    tw = 1.2
    tadd = 0.3

    summary_obj = hhtools.SummaryLoader(fdir_summary, load_only_control=True)

    nadd = int(tadd * srate)
    v_set = uo.collect_chunk(
        cid,
        wid,
        summary_obj=summary_obj,
        target=target,
        srate=srate,
        nequal_len=int(tw * srate),
        nadd=nadd,
        norm=True,
        filt_range=None,
        verbose=verbose,
        tag=tag,
    )

    nlag_max = int(tlag_max * srate / 1000)
    nlag_min = int(tlag_min * srate / 1000)
    nlag_step = int(tlag_step * srate / 1000)

    if method in ("naive", "spo", "mit"):
        fte = tt.compute_te
        params = dict(
            nchunks=nchunks,
            chunk_size=chunk_size,
            nmax_delay=nlag_max,
            nmin_delay=nlag_min,
            nstep_delay=nlag_step,
            method=method,
            nrel_points=list(-np.arange(nhist)),
        )
    elif method == "2d":
        fte = tt.compute_te_2d
        params = dict(
            nchunks=nchunks,
            chunk_size=chunk_size,
            nmax_delay=nlag_max,
            nmin_delay=nlag_min,
            nstep_delay=nlag_step,
        )
    elif method == "full":
        fte = tt.compute_te_full2
        params = dict(
            nchunks=nchunks,
            chunk_size=chunk_size,
            nmax_delay=nlag_max,
            nmin_delay=nlag_min,
            nstep_delay=nlag_step,
        )
    elif method == "embedding":
        fte = tt.compute_te_embedding
        params = dict(
            nchunks=nchunks,
            chunk_size=chunk_size,
            num_emb_dim=num_emb_dim,
            nmax_delay=nlag_max,
            nmin_delay=nlag_min,
            nstep_delay=nlag_step,
        )

    te_true, te_surr, tlag = compute_te(
        fte, v_set, ntrue, nsurr, nadd, params, verbose=verbose
    )

    if fout is None:
        fout = "te_%d%02d.pkl" % (cid, wid)
    if save_dir is not None:
        fout = os.path.join(save_dir, "te_%d%02d.pkl" % (cid, wid))

    with open(fout, "wb") as fp:

        info = {
            "cid": cid,
            "wid": wid,
            "ntrue": ntrue,
            "nsurr": nsurr,
            "tw": tw,
            "tadd": tadd,
            "fdir": fdir_summary,
            "seed": seed,
            "dir": ("0->1", "1->0"),
            "method": method,
            "params": params,
        }

        pkl.dump({"info": info, "tlag": tlag, "te": te_true, "te_surr": te_surr}, fp)


def compute_te(fte, v_set, ntrue, nsurr, nadd, params, verbose=True):

    seed_set = np.random.randint(10, int(1e8), ntrue)

    nsurr_per_chunk = nsurr // ntrue
    f = partial(
        _compute,
        fte=fte,
        v_set=v_set,
        nadd=nadd,
        nsurr_per_chunk=nsurr_per_chunk,
        **params,
    )
    te_results = uo.par_func(f, seed_set, num_process, desc="TE", verbose=verbose)

    tlag = te_results[0][2]
    te_set_true = np.array([te[0] for te in te_results])
    te_set_surr = np.concatenate([te[1] for te in te_results], axis=0)

    return te_set_true, te_set_surr, tlag


def _compute(
    seed,
    fte=None,
    v_set=None,
    nchunks=0,
    nsurr_per_chunk=0,
    chunk_size=0,
    nadd=0,
    **kwargs,
):
    np.random.seed(seed)

    v_sample_true, v_sample_surr = sample_set(
        v_set,
        nchunks=nchunks,
        nsurr_per_chunk=nsurr_per_chunk,
        chunk_size=chunk_size,
        max_delay=kwargs["nmax_delay"],
        nadd=nadd,
    )

    te_true, nlag = fte(v_sample_true, **kwargs)
    te_surr_set = []
    for n in range(nsurr_per_chunk):
        te_surr, _ = fte(v_sample_surr[n], **kwargs)
        te_surr_set.append(te_surr)
    te_surr_set = np.array(te_surr_set)
    tlag = nlag / srate * 1e3

    return te_true, te_surr_set, tlag


def sample_set(
    v_set,
    nchunks: int = 1000,
    nsurr_per_chunk: int = 100,
    chunk_size: int = 100,
    max_delay: int = 0,
    nadd: int = 0,
):

    assert max_delay < nadd

    nlen = chunk_size + max_delay

    v_sample_true = np.zeros((nchunks, 2, nlen))
    v_sample_surr = np.zeros((nsurr_per_chunk, nchunks, 2, nlen))

    n = 0
    refresh = True
    while n < nchunks:

        if refresh:
            v_true = pick_single_sample(v_set, nmax_delay=max_delay, nadd=nadd)
            nmax = v_true.shape[1]
            n0 = np.random.randint(nlen)
            n1 = n0 + nlen
            refresh = False

            if n1 <= nmax:
                v_surr_set = np.array(
                    [transform_surrogate(v_true) for _ in range(nsurr_per_chunk)]
                )

        if n1 <= nmax:
            v_sample_true[n] = v_true[:, n0:n1]
            v_sample_surr[:, n] = v_surr_set[..., n0:n1]

            n0 = n1 - max_delay
            n1 = n0 + nlen
        else:
            n -= 1
            refresh = True

        n += 1

    return v_sample_true, v_sample_surr


def pick_single_sample(v_set, nmax_delay: int = 0, nadd: int = 0):
    idx = np.random.randint(0, len(v_set))
    v_sel = v_set[idx, :, nadd - nmax_delay :]
    return v_sel[:, ~np.isnan(v_sel[0, :])]


def transform_surrogate(v_sub):

    vs_surr = tt.bivariate_surrogates(v_sub[0], v_sub[1])
    return np.array(vs_surr)


if __name__ == "__main__":
    main(**vars(build_args().parse_args()))
