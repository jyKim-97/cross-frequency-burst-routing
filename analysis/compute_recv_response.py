import numpy as np
import argparse
import xarray as xr

from tqdm import tqdm

import os

from pytools import hhtools
from pytools import utils_spk as us

import pdb

"""
Compute recevier neuron response with bootstrapping (for statistical test)
"""

dt = 0.01
num_trans = 40


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", required=True, type=str)
    parser.add_argument(
        "--fdir_out",
        required=True,
        type=str,
        help="Diectory for exporting computed results",
    )
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--wset", nargs="+", type=int)
    parser.add_argument(
        "--cwyaml", type=str, help="yaml file containing the key 'cwpairs'"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tbin", type=float, default=1)
    parser.add_argument(
        "--nsample", type=int, default=100, help="Number of samples for each measure"
    )
    return parser


def main(
    fdir=None,
    fdir_out=None,
    cwyaml=None,
    cid=0,
    wset=None,
    tbin=1,
    seed=42,
    nsample=100,
):
    np.random.seed(seed)

    if cwyaml is not None:
        print("cid and wset will be ignored")

        import yaml

        with open(cwyaml, "r") as fp:
            d = yaml.safe_load(fp)
        cwpairs = d["cwpairs"]
        print(cwpairs)
        cid_set, wid_set = [], []
        cid_prv = -1
        for cid, wid in cwpairs:
            if cid != cid_prv:
                cid_set.append(cid)
                wid_set.append([wid])
                cid_prv = cid
            else:
                wid_set[-1].append(wid)
    else:
        if cid < 1:
            raise ValueError("cid must be larger than 0")
        cid_set = [cid]
        wid_set = [wset]

    for cid, wid_set_sub in zip(cid_set, wid_set):
        compute_response(fdir, fdir_out, cid, wid_set_sub, tbin, nsample)


def compute_response(fdir, fdir_out, cid, wid_set, tbin, nsample):
    summary_obj = hhtools.SummaryLoader(fdir, load_only_control=True)

    print("Cluster %d selected" % (cid))
    print("nw: ", wid_set)

    nc = cid - 1
    winfo = us.load_pickle(os.path.join(fdir, "./osc_motif/motif_info_%d.pkl" % (nc)))[
        "winfo"
    ]
    prob_spk_set, prob_spk_set_p, t = collect_response_sample(
        summary_obj, winfo, nc, wid_set, tbin, nsample
    )

    ds = xr.Dataset(
        data_vars=dict(
            prob=(("nw", "ndelay", "nsample", "ntp", "t"), prob_spk_set),
            prob0=(("nw", "ndelay", "nsample", "ntp", "t"), prob_spk_set_p),
        ),
        coords=dict(
            nw=wid_set,
            ndelay=summary_obj.controls["tdelay_set"],
            nsample=np.arange(nsample),
            ntp=np.array([0, 1]),
            t=t,
        ),
        attrs=dict(
            ntp="0: F(R), 1: S(R)",
            fdir=fdir,
            nc=nc,
            cluster_id=summary_obj.controls["cluster_id"][nc],
            num_trans=num_trans,
        ),
    )

    fout = os.path.join(fdir_out, "prob_spk_set_nc%d_sample.nc" % (nc))

    ds.to_netcdf(fout)

    print("Done, dataset is exported to %s" % (fout))


def collect_response_sample(summary_obj, winfo, nc, nw_set, tbin, nsample):

    detail = summary_obj.load_detail(0, 0, 0)
    tmax = np.round(detail["ts"][-1] * 1e3)
    N = len(detail["step_spk"])

    tspk_edges = np.arange(-5, 40, tbin)
    nspk_edges = (tspk_edges / dt).astype(int)

    num_delay = summary_obj.num_controls[0]
    num_spk_set = np.zeros((len(nw_set), num_delay, nsample, 2, len(nspk_edges) - 1))
    num_spk_set_p = np.zeros((len(nw_set), num_delay, nsample, 2, len(nspk_edges) - 1))
    max_spk = np.zeros((len(nw_set), num_delay, nsample, 2))

    for nd in tqdm(range(num_delay)):
        for nt in range(summary_obj.num_controls[2]):

            osc_idx = np.zeros(int(tmax / dt), dtype=np.int8) - 1
            for i, nw in enumerate(nw_set):
                for nt_w, tl in winfo[nd][nw]:
                    if nt_w != nt:
                        continue
                    t0 = tl[0]
                    t1 = tl[1]
                    n0 = int(t0 / dt * 1e3)
                    n1 = int(t1 / dt * 1e3)
                    assert np.all(osc_idx[n0:n1] == -1)
                    osc_idx[n0:n1] = i

            detail = summary_obj.load_detail(nd, nc, nt)
            sel_tr = read_tr_info(detail["prefix"])
            id_sample = np.random.randint(0, nsample, size=len(sel_tr[0]))

            pseudo_recv_set = [
                list(np.arange(N // 5 * 2)),
                list(np.arange(N // 2, N // 10 * 9)),
            ]
            for i in range(2):
                for idx in sel_tr[i]:
                    ntp = idx // (N // 2)
                    try:
                        pseudo_recv_set[ntp].remove(idx)
                    except:
                        pdb.set_trace()

            for ids, idt, idr in zip(id_sample, sel_tr[0], sel_tr[1]):
                n0 = 0
                ntp = idt // (N // 2)
                for nstep in detail["step_spk"][idt]:
                    if osc_idx[nstep] == -1:
                        continue

                    wid = osc_idx[nstep]
                    num_spk_r, _ = us.align_spike_single(
                        detail["step_spk"][idr], nstep, nspk_edges, nsearch_start=0
                    )
                    num_spk_set[wid, nd, ids, ntp] += num_spk_r

                    idp = np.random.choice(pseudo_recv_set[1 - ntp])
                    num_spk_r, _ = us.align_spike_single(
                        detail["step_spk"][idp], nstep, nspk_edges, nsearch_start=0
                    )
                    num_spk_set_p[wid, nd, ids, ntp] += num_spk_r

                    max_spk[wid, nd, ids, ntp] += 1

    max_spk[max_spk == 0] = 1
    prob_spk_set = num_spk_set / max_spk[..., np.newaxis]
    prob_spk_set_p = num_spk_set_p / max_spk[..., np.newaxis]

    t = (tspk_edges[1:] + tspk_edges[:-1]) / 2

    return prob_spk_set, prob_spk_set_p, t


def read_tr_info(prefix):
    sel_tr = [[], []]
    with open(prefix + "_trinfo.txt", "r") as fp:
        l = fp.readline()
        l = fp.readline()
        while l:
            val = l[:-1].split(",")
            sel_tr[0].append(int(val[0]))
            sel_tr[1].append(int(val[1]))
            l = fp.readline()
    return sel_tr


if __name__ == "__main__":
    main(**vars(build_args().parse_args()))
