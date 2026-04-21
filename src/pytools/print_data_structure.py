import pickle as pkl
import argparse


def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True, type=str)

    return parser


def main(fname=None):

    print(fname)
    with open(fname, "rb") as fp:
        data = pkl.load(fp)

    print("Data %s has" % (fname))
    for k in data.keys():
        print("%s, " % (k), end="")
    print()


if __name__ == "__main__":
    main(**vars(build_arg_parse().parse_args()))
