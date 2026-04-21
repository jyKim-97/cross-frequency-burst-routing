import argparse
import os

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel


def build_args():
    parser = argparse.ArgumentParser(
        description="Compute kappa statistics from sampled receiver responses."
    )
    parser.add_argument("--spk_dir", required=True, type=str)
    parser.add_argument("--cid", required=True, type=int)
    parser.add_argument("--wid", required=True, type=int)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--thres", default=0.0, type=float)
    parser.add_argument("--fdir_out", required=True, type=str)
    parser.add_argument("--sigma", default=1.0, type=float)
    return parser


def load_spk(spk_dir, cid, wid):
    fname = os.path.join(spk_dir, "prob_spk%d.nc" % (cid - 1))
    ds = xr.open_dataset(fname)
    if wid not in ds.nw.data:
        raise ValueError("wid %d does not exist in spike response: %s" % (wid, fname))
    return ds.sel(nw=wid)


def nan_gaussian_filter1d(data, sigma, axis):
    if sigma == 0:
        return data

    finite = np.isfinite(data)
    data_filled = np.where(finite, data, 0.0)
    weight = gaussian_filter1d(finite.astype(float), sigma, axis=axis)
    smoothed = gaussian_filter1d(data_filled, sigma, axis=axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed / weight

    smoothed[weight == 0] = np.nan
    return smoothed


def compute_kappa(spk_wid, spk_wid_base, sigma=1.0, thres=0.0):
    kappa = np.zeros((len(spk_wid.nsample), len(spk_wid.ndelay), len(spk_wid.ntp)))

    for ntp in range(len(spk_wid.ntp)):
        for nd in range(len(spk_wid.ndelay)):
            spk_sub_sample = spk_wid.isel(dict(ndelay=nd, ntp=ntp))
            spk_base_sample = spk_wid_base.isel(dict(ndelay=nd, ntp=ntp))

            dspk = spk_sub_sample.prob - spk_sub_sample.prob0
            dspk_base = spk_base_sample.prob - spk_base_sample.prob0

            dy_base = gaussian_filter1d(dspk_base.mean("nsample").data, sigma)
            id_max = np.argmax(dy_base)
            m = dy_base[id_max]

            if np.abs(m) <= thres:
                kappa[:, nd, ntp] = np.nan
                continue

            dy = gaussian_filter1d(dspk.data, sigma, axis=-1)
            kappa[:, nd, ntp] = (dy[:, id_max] - m) / m

    return nan_gaussian_filter1d(kappa, sigma, axis=1)


def check_stat_trel(kappa, kappa_base):
    p_set, sign_set = [], []

    for nd in range(kappa.shape[1]):
        z1 = kappa[:, nd]
        z2 = kappa_base[:, nd]
        idx = ~(np.isnan(z1) | np.isnan(z2))

        if idx.sum() < 2:
            p_set.append(np.nan)
            sign_set.append(0)
            continue

        result = ttest_rel(z1[idx], z2[idx], alternative="two-sided")
        p_set.append(result.pvalue)

        if np.isfinite(result.statistic):
            sign_set.append(1 if result.statistic > 0 else -1)
        else:
            sign_set.append(0)

    return np.array(p_set), np.array(sign_set)


def main(
    spk_dir=None, cid=None, wid=None, alpha=0.05, thres=0.0, fdir_out=None, sigma=1.0
):
    if cid < 1:
        raise ValueError("cid must be larger than 0")

    os.makedirs(fdir_out, exist_ok=True)

    spk_xa = load_spk(spk_dir, cid, wid)
    spk_xa_base = load_spk(spk_dir, cid, 0)

    kappa = compute_kappa(spk_xa, spk_xa_base, sigma=sigma, thres=thres)
    kappa_base = compute_kappa(spk_xa_base, spk_xa_base, sigma=sigma, thres=thres)

    p_set, sign_set = [], []
    for ntp in range(len(spk_xa.ntp)):
        p_sub, sign_sub = check_stat_trel(kappa[..., ntp], kappa_base[..., ntp])
        p_set.append(p_sub)
        sign_set.append(sign_sub)

    p_set = np.transpose(p_set)
    sign_set = np.transpose(sign_set)
    sig_set = (p_set < alpha) & np.isfinite(p_set)
    signed_sig_set = sig_set.astype(np.int8) * sign_set.astype(np.int8)

    ds = xr.Dataset(
        data_vars=dict(
            kappa=(("nsample", "ndelay", "ntp"), kappa),
            kappa_base=(("nsample", "ndelay", "ntp"), kappa_base),
            pval=(("ndelay", "ntp"), p_set),
            sign=(("ndelay", "ntp"), sign_set),
            sig=(("ndelay", "ntp"), signed_sig_set),
        ),
        coords=dict(
            nsample=spk_xa.nsample.data,
            ndelay=spk_xa.ndelay.data,
            ntp=spk_xa.ntp.data,
        ),
        attrs=dict(
            spk_dir=spk_dir,
            cid=cid,
            wid=wid,
            base_wid=0,
            alpha=alpha,
            thres=thres,
            sigma=sigma,
            source="compute_kappa.py",
            ntp=spk_xa.attrs.get("ntp", "0: F(R), 1: S(R)"),
        ),
    )

    fout = os.path.join(fdir_out, "kappa_%d%02d.nc" % (cid, wid))
    ds.to_netcdf(fout)
    print("Done, dataset is exported to %s" % fout)


if __name__ == "__main__":
    main(**vars(build_args().parse_args()))
