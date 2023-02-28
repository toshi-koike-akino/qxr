# prep_redweb.py
# see: https://sites.google.com/site/redwebcvpr18/
import argparse
import os
import numpy as np

# import scipy.io
from PIL import Image
import gdown

id = "12IjUC6eAiLBX67jW57YQMNRVqUGvTZkX"


def get_args():
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument(
        "--depth",
        default="ReDWeb_V1/RDs",
        type=str,
        help="depth data directory",
    )
    parser.add_argument(
        "--file", default="data/redweb.npz", type=str, help="data npz file"
    )
    return parser.parse_args()


# load EDEN data in a directory
def load_raw(dir):
    deps = []
    rgbs = []
    for fname in sorted(os.listdir(dir)):
        base, ext = os.path.splitext(fname)
        if ext == ".png":
            # loading depth file
            depth_file = os.path.join(dir, fname)
            print(depth_file)
            dep = np.array(Image.open(depth_file))
            print(dep.shape, dep.dtype)

            # loading RGB file matching the depth file name
            rgb_file = os.path.join(dir.replace("RDs", "Imgs", 1), base + ".jpg")
            print(rgb_file)
            try:
                rgb = np.array(Image.open(rgb_file))
            finally:
                # if loaded successfully, then register into list
                print(rgb.shape, rgb.dtype)
                deps.append(dep)
                rgbs.append(rgb)

    # deps = np.stack(deps)
    # rgbs = np.stack(rgbs)
    # print(deps.shape, rgbs.shape)
    print(len(deps), len(rgbs))
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
    elif os.path.exists(os.path.join(args.root, args.depth)):
        # load raw data
        print("loading raw data")
        data = load_raw(os.path.join(args.root, args.depth))
        # archive npz
        archive_data(args.file, data)
    else:
        download(args.root)
        # load raw data
        print("loading raw data")
        data = load_raw(os.path.join(args.root, args.depth))
        # archive npz
        archive_data(args.file, data)

    return data


def download(root):
    os.makedirs(root, exist_ok=True)
    fname = os.path.join(root, "ReDWeb_V1.tar.gz")
    gdown.download(id=id, output=fname, quiet=False)
    gdown.extractall(fname)


def main():
    args = get_args()
    print(args)

    # download(args.root)
    # data = load_raw(args)
    # archive_data(args, data)
    deps, rgbs = load_data(args)
    print(deps.shape, rgbs.shape)


if __name__ == "__main__":
    main()
