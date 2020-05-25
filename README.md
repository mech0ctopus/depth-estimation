# depth-estimation
Practical Depth Estimation with Image Segmentation and Serial U-Nets

![Depth Estimate](depth_estimate.PNG)

![Car Depth Estimate](kitti.gif)

**Depth Estimates on KITTI Validation Data**

```
depth-estimation
|   depth_estimation_nunet.py <--- main file
|   depth_estimate.png
|   inference_timer.py
|   kitti.gif
|   prediction_comparison.py
|   README.md
|   requirements.txt
|
+---data_extraction
|       pickle_kitti_dataset.py
|       pickle_nyu_dataset.py
|       save_to_file_nyu.m
|
+---models
|       losses.py
|       models.py
|
\---utils
        augmented_data_generator.py
        deep_utils.py
        fill_depth_colorization.py
        image_utils.py
        images_2_video.py
        rgb2depth.py
        rgb2depth_stream.py
        stack_videos.py

```

### Initial Setup
```
git clone https://github.com/mech0ctopus/depth-estimation.git
cd depth-estimation
pip install -r requirements.txt
```

### Use Pre-Trained Network on Webcam
1. Download & extract pre-trained weights from link below. Place in depth-estimation folder.
2. Run rgb2depth_stream.
```
cd depth-estimation
python utils\rgb2depth_stream.py
```

### Use Pre-Trained Network on RGB Video
1. Download & extract pre-trained weights from link below. Place in depth-estimation folder.
2. Run video_depth_writer.
```
cd depth-estimation
python utils\video_depth_writer.py
```

###  Train Depth Estimation Network
1. Download NYU Depth V2 or KITTI images from link below
2. (Optional, for NYU Depth V2) Colorize depth images
```
python utils\fill_depth_colorization.py
```
3. Update training & validation folderpaths
4. Verify input shapes are correct (NYU: 480x640, Re-sized KITTI: 192x640)
```
python depth_estimation_nunet.py
```
5. View Results in Tensorboard.
```
cd depth-estimation
tensorboard --logdir logs
```

### Pre-trained Weights
- [Download Pre-trained Weights (NYU Depth V2, ResNet34 Backbones, 480x640 Images)](https://mega.nz/#!y9E1lC7S!UATGE-izPvmzfm_bWeGTkPb9tmoAS8pP4P72iyTQ2pQ)

- [Download Pre-trained Weights (KITTI, ResNet50 Backbones, 192x640 Images)](https://mega.nz/file/L8kHRZSQ#sbZyujgm9CUJL1vdw9D4L6JtTLfS7IzoLtT7mDzI63I)

### Download Pre-processed KITTI Dataset
[Download Pre-processed KITTI RGB and Depth Images (Re-sized and colorized) Training Images (5.5GB)](https://mega.nz/file/O1sn3TQQ#fbXlhG5T8Ad30CTtfwvKyKfgDyH3Aa2tq_fSoYhTA0U)

**Note:** Raw image data is from the [KITTI Raw Dataset (synced and rectified)](http://www.cvlibs.net/datasets/kitti/raw_data.php) and the [KITTI Depth Prediction Dataset (annotated depth maps)](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).

### Datasets
- [FieldSAFE](https://vision.eng.au.dk/fieldsafe/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

### Citation
```
@conference{vehits20,
author={Kyle J. Cantrell. and Craig D. Miller. and Carlos W. Morato.},
title={Practical Depth Estimation with Image Segmentation and Serial U-Nets},
booktitle={Proceedings of the 6th International Conference on Vehicle Technology and Intelligent Transport Systems - Volume 1: VEHITS,},
year={2020},
pages={406-414},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0009781804060414},
isbn={978-989-758-419-0},
}
```