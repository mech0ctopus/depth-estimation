# depth-estimation
Depth Estimation from a Monovision Camera using a Deep Neural Network

```
depth-estimation
|   depth_estimation_nunet.py <--- main file
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
|       unet.py
|
+---utils
|       deep_utils.py
|       image_utils.py
|
\---visualization
        conv_visualization.py
```
### Instructions
```
git clone https://github.com/mech0ctopus/depth-estimation.git
pip install -r requirements.txt
python depth_estimation_nunet.py #Train & evaluate depth estimation neural network
```

### Dataset
- [FieldSAFE](https://vision.eng.au.dk/fieldsafe/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
