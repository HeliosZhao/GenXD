# <div align="center"> GenXD: Generating Any 3D and 4D Scenes <div> 
### <div align="center"> ICLR 2025 <div>
<div align="center">
  <a href="https://gen-x-d.github.io/"><img src="https://img.shields.io/static/v1?label=GenXD&message=Project&color=purple"></a> &ensp;
  <a href="https://arxiv.org/abs/2411.02319"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/datasets/Yuyang-z/CamVid-30K"><img src="https://img.shields.io/static/v1?label=CamVid-30K&message=HuggingFace&color=yellow"></a> &ensp;
</div>

<br>



https://github.com/user-attachments/assets/4ba8b8cd-64cd-4c3d-ad68-eb1a475ca400



## Abstract
> Recent developments in 2D visual generation have been remarkably successful. However, 3D and 4D generation remain challenging in real-world applications due to the lack of large-scale 4D data and effective model design. In this paper, we propose to jointly investigate general 3D and 4D generation by leveraging camera and object movements commonly observed in daily life. Due to the lack of real-world 4D data in the community, we first propose a data curation pipeline to obtain camera poses and object motion strength from videos. Based on this pipeline, we introduce a large-scale real-world 4D scene dataset: CamVid-30K. By leveraging all the 3D and 4D data, we develop our framework, GenXD, which allows us to produce any 3D or 4D scene. We propose multiview-temporal modules, which disentangle camera and object movements, to seamlessly learn from both 3D and 4D data. Additionally, GenXD employs masked latent conditions to support a variety of conditioning views. GenXD can generate videos that follow the camera trajectory as well as consistent 3D views that can be lifted into 3D representations. We perform extensive evaluations across various real-world and synthetic datasets, demonstrating GenXD's effectiveness and versatility compared to previous methods in 3D and 4D generation.

## News
- [26/03/2025] The pre-trained model and inference pipeline are released!
- [24/01/2025] GenXD is accepted to ICLR 2025! We are polishing the code and will release it soon.
- [01/12/2024] CamVid-30K dataset is released!



## TODO
- [x] Release the CamVid-30K dataset.
- [x] Release the pre-trained model.

## Installation

## Inference

### Model Download
Pre-trained model is available on [HuggingFace Model](https://huggingface.co/Yuyang-z/genxd). You can also use the following command to download:
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download Yuyang-z/genxd --local-dir ./genxd-model
```

### Custom Image 3D Generation
Put the images (png or jpg) in the `$DATA_ROOT` directory, choose the camere mode from `forward` and `orbit`, and set the camera parameters to control the cameras. 
```
DATA_ROOT="example-images"
OUTPUT_DIR="outputs/example-images"

accelerate launch --main_process_port 1223 inference.py \
diffusion.pretrain_unet="./genxd-model" \
diffusion.output_dir="$OUTPUT_DIR" \
+evaluator.data_name="static_cam_single" \
+evaluator.data_root="$DATA_ROOT" \
+evaluator.camera_info.mode="forward" +evaluator.camera_info.elevation=0. "+evaluator.camera_info.azimuth_range=[-30,30]" \
+evaluator.focal=1.0938 +evaluator.camera_info.radius=2.0
```

### Few-shot 3D Scene Generation
We follow the ReconFusion data split for few-shot 3D scene generation. The data is packed in the torch format, and it can be downloaded from [HuggingFace Dataset - ReconFusion Data](https://huggingface.co/datasets/Yuyang-z/reconfusion-torch).

In pactice, as we have no idea what the test views are, we can first fit a possible trajectory based on the known few-shot views:
```
# First step: generate pose for each scene
python tools/pose_traj_generate.py -d data/reconfusion-torch/re10k -o outputs/pose_dataset --save_data_pose -sx 0.2 0.4 -0.2 -0.4 -sz 0.2 0.4 -0.2 -0.4 -n 18
```
Subsequently, GenXD can generate views for the fitted trajectory:
```
# Second step: generate reconfusion data with grouped views
accelerate launch --main_process_port 1224 inference.py \
diffusion.pretrain_unet="./genxd-model" \
diffusion.output_dir="outputs/re10k-group" \
+evaluator.data_name="reconfgroup" \
+evaluator.data_root="data/reconfusion-torch/re10k" \
+evaluator.pose_dir="outputs/pose_dataset/re10k" \
+evaluator.num_context_views=3 +evaluator.n_views=3 \
+evaluator.save_target_only=True +evaluator.pad_to_square=True
```

Instead, if we already know what the test views are, we can directly use GenXD to generate the target views:
```
# Optional: directly generate reconfusion target views
accelerate launch --main_process_port 1224 inference.py \
diffusion.pretrain_unet="./genxd-model" \
diffusion.output_dir="outputs/re10k-direct" \
+evaluator.data_name="reconfdirectgen" \
+evaluator.data_root="data/reconfusion-torch/re10k" \
+evaluator.num_context_views=3 +evaluator.n_views=3 \
+evaluator.save_target_only=True +evaluator.pad_to_square=True
```

## CamVid-30K Dataset

CamVid-30K is the first open-sourced, large-scale 4D dataset, designed to support various dynamic 3D tasks. It includes videos sourced from [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset), [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M), and [WebVid-10M](https://github.com/m-bain/webvid), with camera annotations curated using our data curation pipeline.  

### Download
VIPSeg and OpenVid subsets can be downloaded from [HuggingFace Dataset - CamVid-30K](https://huggingface.co/datasets/Yuyang-z/CamVid-30K). 

**Note:** Due to the [ban on WebVid-10M](https://github.com/m-bain/webvid?tab=readme-ov-file#dataset-no-longer-available-but-you-can-still-use-it-for-internal-non-commerical-purposes), we cannot provide the data containing videos from WebVid-10M. If you would like to discuss this subset further, feel free to reach out to [Yuyang](mailto:yuyangzhao98@gmail.com).

### Data Structure

The zipped dataset is organized in the following structure:
```
DATA_PATH
└─ camvid-vipseg
   └─ batch_1.zip
   └─ batch_2.zip
└─ camvid-openvid
   └─ batch_1.zip
   └─ batch_2.zip
   └─ ...
```

After unzipping, each sample contains the images and COLMAP source files:

```
VIDEO_ID
└─ images
   └─ *.jpg
└─ 0
   └─ cameras.bin
   └─ images.bin
   └─ points3D.bin
   └─ project.ini
```




## Citation
If you make use of our work, please cite our paper.
```bibtex
@inproceedings{zhao2024genxd,
  author={Zhao, Yuyang and Lin, Chung-Ching and Lin, Kevin and Yan, Zhiwen and Li, Linjie and Yang, Zhengyuan and Wang, Jianfeng and Lee, Gim Hee and Wang, Lijuan},
  title={GenXD: Generating Any 3D and 4D Scenes},
  booktitle={ICLR},
  year={2025}
}
```

## Acknowledgements
This work is based on numerous outstanding research efforts and open-source contributions, including but not limited to [pixelsplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [SVD Xtend](https://github.com/pixeli99/SVD_Xtend), [ParticleSfM](https://github.com/bytedance/particle-sfm) and [Diffusion4D](https://github.com/VITA-Group/Diffusion4D). Furthermore, we would like to thank Dejia Xu and Yuyang Yin for their valuable discussions on the 4D data.
