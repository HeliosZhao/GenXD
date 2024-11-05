# GenXD

This repository is the official implementation of **GenXD**.

**[GenXD: Generating Any 3D and 4D Scenes](https://arxiv.org/abs/2411.02319)**
<br/>
[Yuyang Zhao](https://yuyangzhao.com), [Chung-Ching Lin](https://www.microsoft.com/en-us/research/people/chunglin/), [Kevin Lin](https://sites.google.com/site/kevinlin311tw/me), [Zhiwen Yan](https://jokeryan.github.io/about/), [Linjie Li](https://www.microsoft.com/en-us/research/people/linjli/), [Zhengyuan Yang](https://zyang-ur.github.io/), [Jianfeng Wang](https://jianfengwang.me/), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/), [Lijuan Wang](https://www.microsoft.com/en-us/research/people/lijuanw/)
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://gen-x-d.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2411.02319-b31b1b.svg)](https://arxiv.org/abs/2411.02319)






https://github.com/user-attachments/assets/a040d25d-718b-4869-8c5a-47f814f33255






## Abstract
> Recent developments in 2D visual generation have been remarkably successful. However, 3D and 4D generation remain challenging in real-world applications due to the lack of large-scale 4D data and effective model design. In this paper, we propose to jointly investigate general 3D and 4D generation by leveraging camera and object movements commonly observed in daily life. Due to the lack of real-world 4D data in the community, we first propose a data curation pipeline to obtain camera poses and object motion strength from videos. Based on this pipeline, we introduce a large-scale real-world 4D scene dataset: CamVid-30K. By leveraging all the 3D and 4D data, we develop our framework, GenXD, which allows us to produce any 3D or 4D scene. We propose multiview-temporal modules, which disentangle camera and object movements, to seamlessly learn from both 3D and 4D data. Additionally, GenXD employs masked latent conditions to support a variety of conditioning views. GenXD can generate videos that follow the camera trajectory as well as consistent 3D views that can be lifted into 3D representations. We perform extensive evaluations across various real-world and synthetic datasets, demonstrating GenXD's effectiveness and versatility compared to previous methods in 3D and 4D generation.

## TODO
- [ ] Release the pre-trained model.
- [ ] Release the CamVid-30K dataset.



## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{zhao2024genxd,
  author={Zhao, Yuyang and Lin, Chung-Ching and Lin, Kevin and Yan, Zhiwen and Li, Linjie and Yang, Zhengyuan and Wang, Jianfeng and Lee, Gim Hee and Wang, Lijuan},
  title={GenXD: Generating Any 3D and 4D Scenes},
  journal={arXiv preprint arXiv:2411.02319},
  year={2024}
}
```

## Acknowledgements
This work is based on numerous outstanding research efforts and open-source contributions, including but not limited to [pixelsplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [SVD Xtend](https://github.com/pixeli99/SVD_Xtend), [ParticleSfM](https://github.com/bytedance/particle-sfm) and [Diffusion4D](https://github.com/VITA-Group/Diffusion4D). Furthermore, we would like to thank Dejia Xu and Yuyang Yin for their valuable discussions on the 4D data.
