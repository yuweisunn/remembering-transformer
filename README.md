# Remembering Transformer
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE) [![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.14-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

The code repository for "Remembering Transformer for Continual Learning" [paper](https://arxiv.org/abs/2404.07518) in PyTorch. 

<p align="center">
<img src="rt.png" width="80%"/>
</p>

## Training

### Split task
      python main_lora_pretrained_AE_lowdim_consolidation_retraining.py --task_type split --dataset cifar10 --num_task 5 --epochs 50 --seed 0
      
### Permutation task
      python main_lora_pretrained_AE_lowdim_consolidation_retraining.py --task_type permute --dataset mnist --num_task 20 --epochs 200 --seed 0

### Download Tiny Imagenet
      bash tinyimagenet_downloader.sh

## Citation
If this repository is helpful for your research or you want to refer the provided results in this work, you could cite the work using the following BibTeX entry:

```
@article{sun2024remembering,
  title={Remembering Transformer for Continual Learning},
  author={Sun, Yuwei and Sakuma, Jun and Kanai, Ryota},
  journal={arXiv preprint arXiv:2404.07518},
  year={2024}
}
```
