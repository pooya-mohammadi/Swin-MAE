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
from scipy import ndimage

parser = ArgumentParser()
parser.add_argument("--input", default="datasets/CT/coronary")
parser.add_argument("--output", default="datasets/two_d/CT/coronary")
parser.add_argument("--save_npz", action="store_true")
parser.add_argument("--save_jpg", action="store_true")
parser.add_argument("--rotate", action="store_true")
parser.add_argument("--pick", type=int, default=None)

parser.add_argument("--n", default=None, type=int)


# def previous_codes():
#     for filepath in tqdm(DirUtils.list_dir_full_path(args.input, interest_extensions=".gz")):
#         output_dir_path = join(args.output, split(filepath)[-1].replace(".nii.gz", ""))
#         os.system(f"med2image -i {filepath} -d {output_dir_path}")
#
#     array = NIBUtils.get_array(filepath).astype(np.float32)
#     for index in range(array.shape[-1]):
#         img = array[..., index]
#         # img = array[..., index:index + 1]
#         # img = np.clip(img, -1024, 1024)
#         # img = (img - np.min(img) / (np.max(img) - np.min(img)) * 255)
#         # img = Image.fromarray(img[:, :, 0], mode='L')
#         img = Image.fromarray(img, mode='L')
#         img.save(os.path.join(args.output, DirUtils.split_extension(os.path.split(filepath)[-1],
#                                                                     suffix=f"_{index}",
#                                                                     extension=".jpg",
#                                                                     current_extension=".nii.gz")))


def save_npz_func(img_path, output_dir, save_jpg: bool = False, rotate: bool = False):
    filename = split(img_path)[-1].replace(".nii.gz", "")
    img_arr = NIBUtils.get_array(img_path)
    os.makedirs(output_dir, exist_ok=True)
    for index in range(img_arr.shape[-1]):
        img_slice = img_arr[..., index]
        if rotate:
            img_slice = ndimage.rotate(img_slice, 90)
        np.savez(join(output_dir, f"{filename}_{index:04}.npz"), img_slice)
        if save_jpg:
            img = Image.fromarray((img_slice / img_arr.max() * 255).astype(np.uint8), mode='L')
            img.save(os.path.join(output_dir, f"{filename}_{index:04}.jpg"))


def process(filepath, save_npz, save_jpg, rotate, pick):
    filename = split(filepath)[-1].replace(".nii.gz", "").replace("_0000", "")
    output_dir_path = join(args.output, filename)

    if save_npz:
        save_npz_func(filepath, output_dir_path, save_jpg=save_jpg, rotate=rotate)
        ext = ".npz"
    else:
        os.system(f"med2image -i {filepath} -d {output_dir_path}")
        ext = ".jpg"

    samples = DirUtils.list_dir_full_path(output_dir_path, interest_extensions=ext)

    if pick:
        name_space = np.linspace(0, len(samples) - 1, num=pick).astype(np.int32).tolist()
    else:
        name_space = list(range(len(samples)))

    for index, img_path in enumerate(samples):
        if index not in name_space:
            os.remove(img_path)
            continue
        shutil.move(img_path, join(args.output, f"{filename}_{index:04}{ext}"))

    if save_npz and save_jpg:  # This is for testing segmentation!
        ext = ".jpg"
        for index, img_path in enumerate(DirUtils.list_dir_full_path(output_dir_path, interest_extensions=ext)):
            if index not in name_space:
                os.remove(img_path)
                continue
            shutil.move(img_path, join(args.output, f"{filename}_{index:04}{ext}"))
    shutil.rmtree(output_dir_path)


if __name__ == '__main__':
    args = parser.parse_args()
    DirUtils.remove_create(args.output)
    Parallel(n_jobs=os.cpu_count() - 2)(
        delayed(process)(filepath, args.save_npz, args.save_jpg, args.rotate, args.pick) for filepath in
        tqdm(DirUtils.list_dir_full_path(args.input, interest_extensions=".gz")[:args.n]))

# python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/narco_desktop/china_seg/ --output /home/aicvi/projects/Swin-MAE-datasets/labels/ct_coronary --save_npz --n 10
# python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/narco_desktop/china/ --output /home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary --n 10
# python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/imagesTr --output /home/aicvi/projects/Swin-MAE-datasets/images/mri_mm --n 100
# python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/labelsTr --output /home/aicvi/projects/Swin-MAE-datasets/labels/mri_mm --save_npz --n 100
