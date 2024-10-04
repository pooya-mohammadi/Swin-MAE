import os
import shutil
from argparse import ArgumentParser
from os.path import join, split
import numpy as np
from deep_utils import NIBUtils, DirUtils
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
from deep_utils import NIBUtils

parser = ArgumentParser()
parser.add_argument("--input_img", default="/home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary/")
parser.add_argument("--input_seg", default="/home/aicvi/projects/Swin-MAE-datasets/labels/ct_coronary/")
parser.add_argument("--save_npz", action="store_true")

args = parser.parse_args()


def check_seg(seg_files: dict[str, str]) -> dict[str, str]:
    output = dict()
    for k, v in seg_files.items():
        data = np.load(v)['arr_0']
        if len(np.unique(data)) == 1:
            os.remove(v)
            continue
        output[k] = v
    return output


if __name__ == '__main__':
    images = DirUtils.list_dir_full_path(args.input_img, interest_extensions=".jpg", return_dict=True)
    labels = DirUtils.list_dir_full_path(args.input_seg, interest_extensions=".npz", return_dict=True)
    labels = check_seg(labels)
    n = 0
    for i_k, i_v in images.items():
        if i_k not in labels:
            os.remove(i_v)
            n += 1

    images = DirUtils.list_dir_full_path(args.input_img, interest_extensions=".jpg", return_dict=True)
    labels = DirUtils.list_dir_full_path(args.input_seg, interest_extensions=".npz", return_dict=True)
    print(f"Removed {n} samples")
    print(len(labels), len(images))

# python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/narco_desktop/china_seg/ --output /home/aicvi/projects/Swin-MAE-datasets/labels/ct_coronary --save_npz --n 10
# python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/narco_desktop/china/ --output /home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary --n 10
# python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/imagesTr --output /home/aicvi/projects/Swin-MAE-datasets/images/mri_mm --n 100
# python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/labelsTr --output /home/aicvi/projects/Swin-MAE-datasets/labels/mri_mm --save_npz --n 100
