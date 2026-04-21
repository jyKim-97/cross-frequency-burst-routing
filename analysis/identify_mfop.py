import os
import pickle as pkl
import argparse
from tqdm import trange

import sys

sys.path.append("../include/pytools")
import oscdetector as od

"""
Export oscillation motif for transmission line
"""


from pytools import hhtools

fdir_root = "/home/jungyoung/Project/hh_neuralnet/"


def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", required=True, type=str, choices=("", "_mslow", "_mfast")
    )
    parser.add_argument("--cid", required=False, type=int, default=0)
    parser.add_argument("--wbin_t", default=0.5, type=float)
    parser.add_argument("--mbin_t", default=0.01, type=float)
    parser.add_argument("--th", default=75, type=float)
    parser.add_argument("--reverse", default=False, type=bool)
    parser.add_argument(
        "--fout", default=None, type=str, help="output file name (.pkl)"
    )
    parser.add_argument(
        "--nc",
        default=-1,
        type=int,
        help="Use if actual cluster id and index is different",
    )
    return parser


summary_obj = None
num_itr_max = -1


srate = 2000


def read_current_time():
    from datetime import datetime

    now = datetime.now()
    return "%d%02d%02d" % (now.year, now.month, now.day)


def load_amp_range(famp):
    with open(famp, "rb") as fp:
        data = pkl.load(fp)

    print("amp_range_set is updated in %s" % (data["last-updated"]))
    return data["amp_range_set"]


def main(
    tag=None, cid=None, wbin_t=None, mbin_t=None, th=75, reverse=False, fout=None, nc=-1
):
    global summary_obj, num_itr_max

    fdir_summary = os.path.join(fdir_root, "gen_three_pop_samples_repr/data%s" % (tag))
    fname_amp = os.path.join(
        fdir_root, "extract_osc_motif/data/osc_motif%s/amp_range_set.pkl" % (tag)
    )

    print("tag: %s" % (tag))
    print("fdir_summary: %s" % (fdir_summary))
    print("fname_amp: %s" % (fname_amp))

    amp_range_set = load_amp_range(fname_amp)
    summary_obj = hhtools.SummaryLoader(fdir_summary)
    num_itr_max = summary_obj.num_controls[1]

    if cid == 0:
        for nc in range(summary_obj.num_controls[0]):
            fout = "./data/osc_motif%s/motif_info_%d.pkl" % (tag, nc + 1)
            save_motif(fout, nc, amp_range_set[nc], wbin_t, mbin_t, th, reverse)
    else:
        if fout is None:
            fout = "./data/osc_motif%s/motif_info_%d.pkl" % (tag, cid)

        if nc == -1:
            nc = cid - 1
        save_motif(fout, nc, amp_range_set[cid - 1], wbin_t, mbin_t, th, reverse)


def save_motif(fout, nc, amp_range, wbin_t, mbin_t, th, reverse):
    winfo = collect_osc_motif(nc, amp_range, wbin_t, mbin_t, th, reverse=reverse)

    print("Saved into %s" % (fout))
    with open(fout, "wb") as fp:
        pkl.dump(
            {
                "winfo": winfo,
                "metainfo": {
                    "amp_range": amp_range,
                    "wbin_t": wbin_t,
                    "mbin_t": mbin_t,
                    "th": th,
                    "reverse": reverse,
                    "last-updated": read_current_time(),
                },
            },
            fp,
        )


def collect_osc_motif(nc, amp_range, wbin_t, mbin_t, th, reverse=False):
    winfo = [[] for _ in range(16)]
    for i in trange(num_itr_max, desc="detecting oscillation motifs"):
        detail_data = summary_obj.load_detail(nc, i)

        psd_set, fpsd, tpsd = od.compute_stfft_all(
            detail_data, mbin_t=mbin_t, wbin_t=wbin_t
        )
        words = od.compute_osc_bit(
            psd_set[1:],
            fpsd,
            tpsd,
            amp_range,
            q=th,
            min_len=2,
            cat_th=2,
            reversed=reverse,
        )
        osc_motif = od.get_motif_boundary(words, tpsd)

        for motif in osc_motif:
            nw = motif["id"]
            tl = motif["range"]
            winfo[nw].append((i, tl))

    return winfo


if __name__ == "__main__":
    main(**vars(build_arg_parse().parse_args()))
