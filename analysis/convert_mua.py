import numpy as np
import sys
from tqdm import tqdm

sys.path.append("/home/jungyoung/Project/hh_neuralnet/include/pytools")
import hhsignal

import hhtools
import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", type=str)
    return parser


def main(tag):
    summary_obj = hhtools.SummaryLoader("./data" + tag, load_only_control=True)
    for nc in range(summary_obj.num_controls[0]):
        for nt in tqdm(range(summary_obj.num_controls[1]), desc="#%d" % (nc)):
            detail = summary_obj.load_detail(nc, nt)
            export_mua(detail, dt=0.01, st=0.001)


def export_mua(detail, dt=0.01, st=0.001):
    prefix = detail["prefix"]
    fname = prefix + "_mua.dat"
    if os.path.exists(fname):
        return

    mua = hhsignal.get_mua(detail, dt=dt, st=st)
    assert mua.shape[0] == 2
    assert mua.shape[1] == len(detail["vlfp"][0])
    data = np.concatenate([[dt], [st], mua[0], mua[1]]).astype(np.float32)
    data.tofile(fname)


if __name__ == "__main__":
    main(**vars(build_parser().parse_args()))
