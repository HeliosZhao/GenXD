## <div align="center"> GenXD: Generating Any 3D and 4D Scenes <div> 
<div align="center">
  <a href="https://gen-x-d.github.io/"><img src="https://img.shields.io/static/v1?label=GenXD&message=Project&color=purple"></a> &ensp;
  <a href="https://arxiv.org/abs/2411.02319"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/datasets/Yuyang-z/CamVid-30K"><img src="https://img.shields.io/static/v1?label=CamVid-30K&message=HuggingFace&color=yellow"></a> &ensp;
</div>

<br>



https://github.com/user-attachments/assets/4ba8b8cd-64cd-4c3d-ad68-eb1a475ca400



### Abstract
> Recent developments in 2D visual generation have been remarkably successful. However, 3D and 4D generation remain challenging in real-world applications due to the lack of large-scale 4D data and effective model design. In this paper, we propose to jointly investigate general 3D and 4D generation by leveraging camera and object movements commonly observed in daily life. Due to the lack of real-world 4D data in the community, we first propose a data curation pipeline to obtain camera poses and object motion strength from videos. Based on this pipeline, we introduce a large-scale real-world 4D scene dataset: CamVid-30K. By leveraging all the 3D and 4D data, we develop our framework, GenXD, which allows us to produce any 3D or 4D scene. We propose multiview-temporal modules, which disentangle camera and object movements, to seamlessly learn from both 3D and 4D data. Additionally, GenXD employs masked latent conditions to support a variety of conditioning views. GenXD can generate videos that follow the camera trajectory as well as consistent 3D views that can be lifted into 3D representations. We perform extensive evaluations across various real-world and synthetic datasets, demonstrating GenXD's effectiveness and versatility compared to previous methods in 3D and 4D generation.

### News
- [01/12/2024] CamVid-30K dataset is released!


### TODO
- [x] Release the CamVid-30K dataset.
- [ ] Release the pre-trained model.

### CamVid-30K Dataset

CamVid-30K is the first open-sourced, large-scale 4D dataset, designed to support various dynamic 3D tasks. It includes videos sourced from [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset), [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M), and [WebVid-10M](https://github.com/m-bain/webvid), with camera annotations curated using our data curation pipeline.  

#### Download
VIPSeg and OpenVid subsets can be downloaded from [HuggingFace Dataset](https://huggingface.co/datasets/Yuyang-z/CamVid-30K). 

**Note:** Due to the [ban on WebVid-10M](https://github.com/m-bain/webvid?tab=readme-ov-file#dataset-no-longer-available-but-you-can-still-use-it-for-internal-non-commerical-purposes), we cannot provide the data containing videos from WebVid-10M. If you would like to discuss this subset further, feel free to reach out to [Yuyang](mailto:yuyangzhao98@gmail.com).

#### Data Structure

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






### Citation
If you make use of our work, please cite our paper.
```bibtex
@article{zhao2024genxd,
  author={Zhao, Yuyang and Lin, Chung-Ching and Lin, Kevin and Yan, Zhiwen and Li, Linjie and Yang, Zhengyuan and Wang, Jianfeng and Lee, Gim Hee and Wang, Lijuan},
  title={GenXD: Generating Any 3D and 4D Scenes},
  journal={arXiv preprint arXiv:2411.02319},
  year={2024}
}
```

### Acknowledgements
This work is based on numerous outstanding research efforts and open-source contributions, including but not limited to [pixelsplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [SVD Xtend](https://github.com/pixeli99/SVD_Xtend), [ParticleSfM](https://github.com/bytedance/particle-sfm) and [Diffusion4D](https://github.com/VITA-Group/Diffusion4D). Furthermore, we would like to thank Dejia Xu and Yuyang Yin for their valuable discussions on the 4D data.
