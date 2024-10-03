import os
import shutil
from argparse import ArgumentParser
from os.path import join, split
import numpy as np
from deep_utils import NIBUtils, DirUtils
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

parser = ArgumentParser()
parser.add_argument("--input", default="datasets/CT/coronary")
parser.add_argument("--output", default="datasets/two_d/CT/coronary")


def process(filepath):
    filename = split(filepath)[-1].replace(".nii.gz", "")
    output_dir_path = join(args.output, filename)
    os.system(f"med2image -i {filepath} -d {output_dir_path}")
    for index, img_path in enumerate(DirUtils.list_dir_full_path(output_dir_path, interest_extensions=".jpg")):
        shutil.move(img_path, join(args.output, f"{filename}_{index:04}.jpg"))
    shutil.rmtree(output_dir_path)


if __name__ == '__main__':
    args = parser.parse_args()
    DirUtils.remove_create(args.output)
    Parallel(n_jobs=os.cpu_count() - 3)(delayed(process)(filepath, ) for filepath in
                                        tqdm(DirUtils.list_dir_full_path(args.input, interest_extensions=".gz")))
    # for filepath in tqdm(DirUtils.list_dir_full_path(args.input, interest_extensions=".gz")):
    #     output_dir_path = join(args.output, split(filepath)[-1].replace(".nii.gz", ""))
    #     os.system(f"med2image -i {filepath} -d {output_dir_path}")

    # array = NIBUtils.get_array(filepath).astype(np.float32)
    # for index in range(array.shape[-1]):
    #     img = array[..., index]
    #     # img = array[..., index:index + 1]
    #     # img = np.clip(img, -1024, 1024)
    #     # img = (img - np.min(img) / (np.max(img) - np.min(img)) * 255)
    #     # img = Image.fromarray(img[:, :, 0], mode='L')
    #     img = Image.fromarray(img, mode='L')
    #     img.save(os.path.join(args.output, DirUtils.split_extension(os.path.split(filepath)[-1],
    #                                                                 suffix=f"_{index}",
    #                                                                 extension=".jpg",
    #                                                                 current_extension=".nii.gz")))
    # break
