# depth-estimation
Practical Depth Estimation with Image Segmentation and Serial U-Nets

![Depth Estimate](depth_estimate.PNG)


```
depth-estimation
|   depth_estimation_nunet.py <--- main file
|   depth_estimate.png
|   inference_timer.py
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
+---utils
|       data_generator.py
|       deep_utils.py
|       fill_depth_colorization.py
|       image_utils.py
|       video_stream.py
|
\---visualization
        conv_visualization.py
```
### Instructions
```
git clone https://github.com/mech0ctopus/depth-estimation.git
pip install -r requirements.txt
#Download & extract NYU Depth V2 images from link below
#(Optional) Colorize depth images
python utils\fill_depth_colorization.py
#Update training & validation folderpaths
python depth_estimation_nunet.py #Train & evaluate depth estimation neural network

#To run live depth estimation on webcam video stream
#Download & extract pre-trained weights from link below
cd depth-estimation\utils
python video_stream.py
```
### Pre-trained Weights
[Download Pre-trained Weights](https://mega.nz/#!y9E1lC7S!UATGE-izPvmzfm_bWeGTkPb9tmoAS8pP4P72iyTQ2pQ)

### Datasets
- [FieldSAFE](https://vision.eng.au.dk/fieldsafe/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
