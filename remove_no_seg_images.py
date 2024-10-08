import os
import shutil
from argparse import ArgumentParser
from os.path import join, split

import PIL.Image
import numpy as np
from deep_utils import NIBUtils, DirUtils
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
from deep_utils import NIBUtils

parser = ArgumentParser()
parser.add_argument("--input_img", default="/home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary/")
parser.add_argument("--input_seg", default="/home/aicvi/projects/Swin-MAE-datasets/labels/ct_coronary/")
parser.add_argument("--img_ext", default=".jpg", type=str)
parser.add_argument("--seg_ext", default=".jpg", type=str)
parser.add_argument("--remove_empty", action="store_true", help="If set will remove segments without target")

args = parser.parse_args()


def check_seg(seg_files: dict[str, str]) -> dict[str, str]:
    output = dict()
    for k, v in tqdm(seg_files.items(), desc="Checking the segmentation:"):
        data = np.load(v)['arr_0']
        if len(np.unique(data)) == 1:
            os.remove(v)
            continue
        output[k] = v
    return output


if __name__ == '__main__':
    images = DirUtils.list_dir_full_path(args.input_img, interest_extensions=args.img_ext, return_dict=True)
    labels = DirUtils.list_dir_full_path(args.input_seg, interest_extensions=args.seg_ext, return_dict=True)

    if args.remove_empty:
        labels = check_seg(labels)

    n = 0
    for i_k, i_v in tqdm(images.items(), desc="Removing non existing images"):
        if i_k not in labels:
            os.remove(i_v)
            n += 1

    images = DirUtils.list_dir_full_path(args.input_img, interest_extensions=args.img_ext, return_dict=True)
    labels = DirUtils.list_dir_full_path(args.input_seg, interest_extensions=args.seg_ext, return_dict=True)
    print(f"Removed {n} empty samples")
    print(len(labels), len(images))

    # Remove images that don't have the same shape as label
    n = 0
    for img_key, img_value in tqdm(images.items(), desc="Removing non consistent shapes"):
        label_value = labels[img_key]
        label = np.load(label_value)['arr_0'] if label_value.endswith(".npz") else np.array(PIL.Image.open(label_value))
        image = np.array(Image.open(img_value))[:, :, 2]
        if image.shape != label.shape:
            os.remove(img_value)
            os.remove(label_value)
            n += 1

    images = DirUtils.list_dir_full_path(args.input_img, interest_extensions=args.img_ext, return_dict=True)
    labels = DirUtils.list_dir_full_path(args.input_seg, interest_extensions=args.seg_ext, return_dict=True)
    print(f"Removed {n} samples")
    print(len(labels), len(images))

# python remove_no_seg_images.py --input_img /home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary/ --input_seg /home/aicvi/projects/Swin-MAE-datasets/labels/ct_coronary/
# python remove_no_seg_images.py --input_img /home/aicvi/projects/Swin-MAE-datasets/images/mri_mm --input_seg /home/aicvi/projects/Swin-MAE-datasets/labels/mri_mm
