# IS-MVSNet (ECCV 2022)

Our paper has been accepted as a conference paper in ECCV 2022!

ISMVSNet, a.k.a. Importance-sampling-based MVSNet, is a simple yet effective multi-view reconstruction method. 

This repo provides a Mindspore-based implementation of IS-MVSNet. You may **star** and **watch** this repo for further updates.

### Installation
```shell
# Centos 7.9.2009 is recommended.
# CUDA == 11.1, GCC == 7.3.0, Python == 3.7.9
conda create -n ismvsnet python=3.7.9
conda install mindspore-gpu=1.7.0 cudatoolkit=11.1 -c mindspore -c conda-forge  # Install mindspore == 1.7.0
pip install numpy, opencv-python, tqdm, Pillow
conda activate ismvsnet
```

### Pretrained Model
The pretrained weights for the backbone are already placed under `./weights`. The weights for stages 1 to 3 can be downloaded from the [pretrained weights](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lwangcg_connect_ust_hk/EjJEn_0ZGPNBlB3SRpc48b0BfVo3eS4VfCNNxB5LoAAWEQ?e=r0yeF3).

### Data structure
```
DATAROOT
└───data
|   └───tankandtemples
|       └───intermediate
|           └───Playground
|           │   └───rmvs_scan_cams
|           │       │   00000000_cam.txt
|           │       │   00000001_cam.txt
|           │       │   ...
|           │   └───images
|           │       │   00000000.jpg
|           │       │   00000001.jpg
|           │       │   ...
|           │   └───pair.txt
|           │   └───Playground.log
|           └───Family
|           └───...
|       └───advanced
└───weights
└───src
└───validate.py
└───point_cloud_generator.py
```
         
### Depth prediction
```bash
python validate.py
```

The depth predictions will be saved to 'results/{dataset_name}/{split}/depth'

### Point cloud fusion
```bash
python point_cloud_generator.py
```

The fused point clouds will be saved to 'results/{dataset_name}/{split}/points'

### Citation
If you think this repo is helpful, please consider citing our paper:
```
@InProceedings{ismvsnet,
author="Wang, Likang
and Gong, Yue
and Ma, Xinjun
and Wang, Qirui
and Zhou, Kaixuan
and Chen, Lei",
editor="Avidan, Shai
and Brostow, Gabriel
and Ciss{\'e}, Moustapha
and Farinella, Giovanni Maria
and Hassner, Tal",
title="IS-MVSNet:Importance Sampling-Based MVSNet",
booktitle="Computer Vision -- ECCV 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="668--683",
abstract="This paper presents a novel coarse-to-fine multi-view stereo (MVS) algorithm called importance-sampling-based MVSNet (IS-MVSNet) to address a crucial problem of limited depth resolution adopted by current learning-based MVS methods. We proposed an importance-sampling module for sampling candidate depth, effectively achieving higher depth resolution and yielding better point-cloud results while introducing no additional cost. Furthermore, we proposed an unsupervised error distribution estimation method for adjusting the density variation of the importance-sampling module. Notably, the proposed sampling module does not require any additional training and works reasonably well with the pre-trained weights of the baseline model. Our proposed method leads to up to {\$}{\$}20{\backslash}times {\$}{\$}20{\texttimes}promotion on the most refined depth resolution, thus significantly benefiting most scenarios and excellently superior on fine details. As a result, IS-MVSNet outperforms all the published papers on TNT's intermediate benchmark with an F-score of 62.82{\%}. Code is available at github.com/NoOneUST/IS-MVSNet.",
isbn="978-3-031-19824-3"
}
```
