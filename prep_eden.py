# prep_eden.py
import argparse
import os
import numpy as np
import scipy.io
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument(
        "--depth",
        default="data/EDEN/Depth/0001/clear",
        type=str,
        help="depth data directory",
    )
    parser.add_argument(
        "--file", default="data/eden.npz", type=str, help="data npz file"
    )
    return parser.parse_args()


# load EDEN data in a directory
def load_raw(dir):
    deps = []
    rgbs = []
    for fname in sorted(os.listdir(dir)):
        base, ext = os.path.splitext(fname)
        if ext == ".mat":
            # loading depth file
            depth_file = os.path.join(dir, fname)
            print(depth_file)
            dep = scipy.io.loadmat(depth_file)["Depth"]
            print(dep.shape, dep.dtype)

            # loading RGB file matching the depth file name
            rgb_file = os.path.join(dir.replace("Depth", "RGB", 1), base + ".png")
            print(rgb_file)
            try:
                rgb = np.array(Image.open(rgb_file))
            finally:
                # if loaded successfully, then register into list
                print(rgb.shape, rgb.dtype)
                deps.append(dep)
                rgbs.append(rgb)

    deps = np.stack(deps)
    rgbs = np.stack(rgbs)
    print(deps.shape, rgbs.shape)
    return deps, rgbs


def archive_data(file, data):
    print("saveing", file)
    np.savez_compressed(file, Depth=data[0], RGB=data[1])


def load_npz(file):
    # load archived npz
    print("loading", file)
    npz = np.load(file)
    data = (npz["Depth"], npz["RGB"])
    return data


def load_data(args):
    if os.path.exists(args.file):
        data = load_npz(args.file)
    else:
        # load raw data
        print("loading raw data")
        data = load_raw(args.depth)
        # archive npz
        archive_data(args.file, data)
    return data


def main():
    args = get_args()
    print(args)

    # data = load_raw(args)
    # archive_data(args, data)
    deps, rgbs = load_data(args)
    print(deps.shape, rgbs.shape)


if __name__ == "__main__":
    main()
