"""Extract burst when two populations are disconnected"""

import numpy as np
import os
import pickle as pkl
import argparse
from tqdm import tqdm

from pytools import hhsignal, hhtools
from pytools import burst_tools as bt

fdirs = {"fast": "./data/pe_nu_fast", "slow": "./data/pe_nu_slow"}

post_dir = "./postdata"

wbin_t = 0.5
mbin_t = 0.01


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract burst when two populations are disconnected"
    )

    return parser


def main():

    for lb, fdir in fdirs.items():
        summary_obj = hhtools.SummaryLoader(fdir)
        bprops = extract_probs(summary_obj)

        fout = os.path.join(post_dir, "bprops_%s.pkl" % (lb))
        with open(fout, "wb") as fp:
            pkl.dump(
                dict(
                    attrs=dict(fdir=fdir, wbin_t=wbin_t, mbin_t=mbin_t),
                    burst_props=bprops,
                ),
                fp,
            )

        print("File saved into %s" % (fout))


def extract_probs(summary_obj):
    NUM_ROW, NUM_COL, NUM_TRIAL = summary_obj.num_controls

    burst_props = dict(burst_f=[], burst_len=[], burst_amp=[], idx=[])

    i_prv, j_prv = -1, -1
    for n in tqdm(range(NUM_ROW * NUM_COL * NUM_TRIAL)):
        i = n // (NUM_COL * NUM_TRIAL)
        j = (n - i * NUM_COL * NUM_TRIAL) // NUM_TRIAL
        k = n % NUM_TRIAL

        if i != i_prv:
            for key in ("burst_f", "burst_len", "burst_amp", "idx"):
                burst_props[key].append([])
            i_prv = i

        if j != j_prv:
            for key in ("burst_f", "burst_len", "burst_amp", "idx"):
                burst_props[key][-1].append([])
            j_prv = j

        detail = summary_obj.load_detail(i, j, k)
        psd, fpsd, tpsd = hhsignal.get_stfft(
            detail["vlfp"][0],
            detail["ts"],
            2000,
            frange=(10, 90),
            wbin_t=wbin_t,
            mbin_t=mbin_t,
        )

        idf = np.argmax(psd.mean(axis=1))
        th_m = psd.mean(axis=1)[idf]
        th_s = psd.std(axis=1)[idf]

        bmap = bt.find_blob_filtration(
            psd, th_m, th_s, std_min=0.1, std_max=8, std_step=0.1, nmin_width=-1
        )
        burst_f, burst_range, burst_amp = bt.extract_burst_attrib(psd, fpsd, bmap)

        burst_props["burst_f"][-1][-1].extend(burst_f)
        burst_props["burst_len"][-1][-1].extend(burst_range[:, 1] - burst_range[:, 0])
        burst_props["burst_amp"][-1][-1].extend(burst_amp)
        burst_props["idx"][-1][-1].extend([n] * len(burst_f))


if __name__ == "__main__":
    main(**vars(build_parser().parse_args()))
