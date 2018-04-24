## Prerequisites

1. FFMPEG
2. MatConvNet
3. For face alignment
  (a) scikit-image
  (b) dlib 18.18 (install using setup.py)
  (c) numpy
  (d) scipy

## Setting up Matconvnet
```
cd matconvnet_gen
addpath matlab
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-8.0', ...
               'cudaMethod', 'nvcc') ;
vl_testnn('gpu', true)
```
## Demo

Download model:
```
sh download_model.sh
```

Run demo in MATLAB:
```
run_demo
```

## Face alignment
```
cd face_detection
y = baseface('../data/obama.jpg');
```
5 aligned identity images are then concatenated channel-wise. See faceimg.mat for examples.


## Citation
Please cite the paper below if you make use of the demo. 
```
@InProceedings{Chung17b,
  author       = "Chung, J.~S. and Jamaludin, A. and Zisserman, A.",
  title        = "You said that?",
  booktitle    = "British Machine Vision Conference",
  year         = "2017",
}
```
