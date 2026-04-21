import numpy as np
import xarray as xa
from tqdm import tqdm
from tdigest import TDigest

import sys

sys.path.append("../include/pytools")
import hhtools
import hhsignal
import argparse

axis_set = ((0, 0), (0, 1), (1, 1))


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, default=1, help="Cluster ID")
    parser.add_argument("--mbin_t", type=float, default=0.01)
    parser.add_argument(
        "--wbin_t", type=float, default=0.5, help="Window length of the FFT in seconds"
    )
    parser.add_argument("--nperm_ratio", type=int, default=10)
    parser.add_argument(
        "--perm_prt", type=int, default=90, help="Percentile for permutation threshold"
    )
    parser.add_argument("--th_std", type=float, default=None)
    parser.add_argument("--th_prt", type=float, default=None)
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Path to data directory"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./",
        help="Optional path to save co_map as .nc file (xarray)",
    )
    parser.add_argument("--fmin", type=float, default=10)
    parser.add_argument("--fmax", type=float, default=90)
    parser.add_argument("--df", type=float, default=2)
    parser.add_argument("--fwin", type=float, default=5)
    return parser


def main(
    cid=1,
    mbin_t=0.01,
    wbin_t=0.5,
    th_std=1.96,
    th_prt=90,
    fmin=10,
    fmax=90,
    df=2,
    fwin=5,
    nperm_ratio=10,
    perm_prt=90,
    data_dir="./data",
    save_path="./",
):
    if th_std is None and th_prt is None:
        raise ValueError("Either th_std or th_prt must be specified.")
    elif th_std is not None and th_prt is not None:
        raise ValueError("Only one of th_std or th_prt should be specified.")

    pjoint_mean, pjoint_var, pjoint_th, fpsd = compute_co_map(
        cid=cid,
        mbin_t=mbin_t,
        wbin_t=wbin_t,
        df=df,
        fwin=fwin,
        flim=(fmin, fmax),
        nperm_ratio=nperm_ratio,
        perm_prt=perm_prt,
        th_std=th_std,
        th_prt=th_prt,
        data_dir=data_dir,
    )

    data = xa.DataArray(
        np.array([pjoint_mean, pjoint_var, pjoint_th]),
        coords=dict(
            mv=["mean", "var", "thr"], type=["ff", "fs", "ss"], f1=fpsd, f2=fpsd
        ),
        attrs=dict(
            description="Co-occurrence map of PSD bursts across two LFP channels",
            cid=cid,
            mbin_t=mbin_t,
            wbin_t=wbin_t,
            th_std=th_std if th_std is not None else -1,
            th_prt=th_prt if th_prt is not None else -1,
            perm_prt=perm_prt,
            nperm_ratio=nperm_ratio,
            data_dir=data_dir,
        ),
    )

    fout = f"{save_path}/co_map_{cid}.nc"
    data.to_netcdf(fout)
    print(f"Co-occurrence map saved to {fout}")


def compute_co_map(
    cid=1,
    mbin_t=0.01,
    wbin_t=0.5,
    th_std=2.58,
    th_prt=90,
    nperm_ratio=10,
    perm_prt=90,
    data_dir="./data",
    df=2,
    fwin=5,
    flim=(10, 90),
):
    """
    Compute co-occurrence map of thresholded PSD bursts across two LFP channels.

    Args:
        cid (int): Condition ID (1-based index for summary_obj).
        mbin_t (float): Time resolution of the short-time FFT.
        wbin_t (float): Window length of the FFT in seconds.
        th_std (float): Threshold for binarizing PSD using mean + th_std * std.
        data_dir (str): Path to directory containing the data.

    Returns:
        np.ndarray: Co-occurrence map of PSD bursts (shape: NT x fdim x fdim).
    """
    summary_obj = hhtools.SummaryLoader(data_dir, load_only_control=True)
    NT = summary_obj.num_controls[1]

    pjoint_true_mean = None
    pjoint_true_var = None
    tdigest_set = None

    fedges = np.array(
        [[f0 - fwin, f0 + fwin] for f0 in np.arange(flim[0], flim[1] + df / 2, df)]
    )

    for nt in tqdm(range(NT), desc="Computing co_map"):

        detail = summary_obj.load_detail(cid - 1, nt)
        psd_set, fpsd, tpsd = get_stfft_all(
            [detail["vlfp"][i] for i in range(1, 3)],
            detail["ts"],
            2000,
            mbin_t=mbin_t,
            wbin_t=wbin_t,
        )

        psd_set, fpsd = coarse_grain_fpsd(psd_set, fpsd, fedges)

        if th_prt is not None:
            psd_th = np.percentile(psd_set, th_prt, axis=-1)
        if th_std is not None:
            psd_th = np.mean(psd_set, axis=-1) + th_std * np.std(psd_set, axis=-1)
        psd_bin = (psd_set > psd_th[:, :, None]).astype(float)

        if pjoint_true_mean is None:
            pjoint_true_mean = np.zeros((3, len(fpsd), len(fpsd)))
            pjoint_true_var = np.zeros_like(pjoint_true_mean)
            tdigest_set = init_tdigest(fpsd)

        for nax, (i, j) in enumerate(axis_set):
            x, y = psd_bin[i], psd_bin[j]

            pjoint_true = get_joint(x, y)
            pjoint_true_mean[nax] += pjoint_true
            pjoint_true_var[nax] += pjoint_true**2

            for k in range(nperm_ratio):
                xp = permutate(x)
                yp = y
                pjoint_perm = get_joint(xp, yp)
                update_tdigest(tdigest_set[nax], pjoint_perm)

    for i in range(3):
        pjoint_true_mean[i] /= NT
        pjoint_true_var[i] /= NT
        pjoint_true_var[i] -= pjoint_true_mean[i] ** 2

    pjoint_th = np.zeros_like(pjoint_true_mean)
    for nax in range(3):
        pjoint_th[nax] = percentile_tdigest(tdigest_set[nax], prt=perm_prt)

    return pjoint_true_mean, pjoint_true_var, pjoint_th, fpsd


def init_tdigest(fpsd):

    td_set = []
    for i in range(3):
        td_set.append([])
        for n1 in range(len(fpsd)):
            td_set[-1].append([])
            for n2 in range(len(fpsd)):
                if i != 1 and n2 < n1:
                    td_set[-1][-1].append(None)
                else:
                    td_set[-1][-1].append(TDigest())
    return td_set


def update_tdigest(td_set, pjoint_perm):
    N = len(td_set)
    for n1 in range(N):
        for n2 in range(N):
            if td_set[n1][n2] is None:
                continue
            td_set[n1][n2].update(pjoint_perm[n1, n2], w=1)


def percentile_tdigest(td_set, prt=90):
    N = len(td_set)
    prt_set = np.zeros((N, N))
    for n1 in range(N):
        for n2 in range(N):
            if td_set[n1][n2] is None:
                prt_set[n1, n2] = np.nan
            else:
                prt_set[n1, n2] = td_set[n1][n2].percentile(prt)

    for n1 in range(N):
        for n2 in range(N):
            if np.isnan(prt_set[n1, n2]):
                prt_set[n1, n2] = prt_set[n2, n1]

    return prt_set


def coarse_grain_fpsd(psd_set, fpsd, fedges):

    psd2 = np.zeros((psd_set.shape[0], len(fedges), psd_set.shape[2]))
    for nf in range(len(fedges)):
        f0, f1 = fedges[nf]
        idf = (fpsd >= f0) & (fpsd < f1)
        psd2[:, nf, :] = psd_set[:, idf, :].mean(axis=1)
    fpsd2 = np.mean(fedges, axis=1)

    return psd2, fpsd2


def get_stfft_all(vlfp_set, t, fs, teq=0.5, mbin_t=0.01, wbin_t=0.5, frange=(5, 100)):
    vlfp_set = np.array(vlfp_set)
    idt = t >= teq

    psd_set = []
    for i in range(vlfp_set.shape[0]):
        psd, fpsd, tpsd = hhsignal.get_stfft(
            vlfp_set[i][idt], t[idt], fs, mbin_t=mbin_t, wbin_t=wbin_t, frange=frange
        )
        psd_set.append(psd)
    return np.array(psd_set), fpsd, tpsd


def get_joint(x2, y2):

    T = x2.shape[1]
    p = np.dot(x2, y2.T) / T
    return p


def permutate(x):
    xp = np.zeros_like(x)
    for i in range(xp.shape[0]):
        xp[i] = np.random.permutation(x[i])
    return xp


if __name__ == "__main__":
    main(**vars(build_args().parse_args()))
