python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/narco_desktop/china_seg/ --output /home/aicvi/projects/Swin-MAE-datasets/labels/ct_coronary --save_npz --n 10
python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/narco_desktop/china/ --output /home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary --n 10
python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/imagesTr --output /home/aicvi/projects/Swin-MAE-datasets/images/mri_mm --n 100
python convert_3D_to_2D.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/labelsTr --output /home/aicvi/projects/Swin-MAE-datasets/labels/mri_mm --save_npz --n 100

python remove_no_seg_images.py --input_img /home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary/ --input_seg /home/aicvi/projects/Swin-MAE-datasets/labels/ct_coronary/
python remove_no_seg_images.py --input_img /home/aicvi/projects/Swin-MAE-datasets/images/mri_mm --input_seg /home/aicvi/projects/Swin-MAE-datasets/labels/mri_mm