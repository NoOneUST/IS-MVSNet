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
