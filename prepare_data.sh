python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/m40e-china/train/labels/ --output /home/aicvi/projects/Swin-MAE-datasets/train/labels/ct_coronary/ --save_npz --rotate --pick 10
python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/m40e-china/train/images/ --output /home/aicvi/projects/Swin-MAE-datasets/train/images/ct_coronary/ --pick 10

python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/m40e-china/val/labels/ --output /home/aicvi/projects/Swin-MAE-datasets/val/labels/ct_coronary/ --save_npz --rotate --pick 10
python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/m40e-china/val/images/ --output /home/aicvi/projects/Swin-MAE-datasets/val/images/ct_coronary/ --pick 10

python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/imagesTr --output /home/aicvi/projects/Swin-MAE-datasets/train/images/mri_mm --pick 5
python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/labelsTr --output /home/aicvi/projects/Swin-MAE-datasets/train/labels/mri_mm --save_npz --rotate --pick 5

python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/imagesTs --output /home/aicvi/projects/Swin-MAE-datasets/val/images/mri_mm --pick 5
python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/labelsTs --output /home/aicvi/projects/Swin-MAE-datasets/val/labels/mri_mm --save_npz --rotate --pick 5


python remove_no_seg_images.py --input_img /home/aicvi/projects/Swin-MAE-datasets/train/images/ct_coronary/ --input_seg /home/aicvi/projects/Swin-MAE-datasets/train/labels/ct_coronary/
python remove_no_seg_images.py --input_img /home/aicvi/projects/Swin-MAE-datasets/train/images/mri_mm/ --input_seg /home/aicvi/projects/Swin-MAE-datasets/train/labels/mri_mm/

python remove_no_seg_images.py --input_img /home/aicvi/projects/Swin-MAE-datasets/val/images/ct_coronary/ --input_seg /home/aicvi/projects/Swin-MAE-datasets/val/labels/ct_coronary/
python remove_no_seg_images.py --input_img /home/aicvi/projects/Swin-MAE-datasets/val/images/mri_mm/ --input_seg /home/aicvi/projects/Swin-MAE-datasets/val/labels/mri_mm/