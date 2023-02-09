# A Simple Plugin for Transforming Images to Arbitrary Scales

This repository contains the official PyTorch implementation of ARIS: https://arxiv.org/abs/2210.03417.

## Requirements
A suitable [conda](https://conda.io/) environment named `aris` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate aris
```

## Model Zoo

https://drive.google.com/drive/folders/1GmZiTC5NSx-UnT0SwPeJQqjWtThc6yh-?usp=sharing

## Train

Comming soon....

## Test

Test set: https://drive.google.com/drive/folders/1GmZiTC5NSx-UnT0SwPeJQqjWtThc6yh-?usp=sharing

```bash
python test_final.py --model hat+aris \
--dir_data 'xxx/dataset' \
--pretrain 'xxx/HAT_ARIS.pt' \
--data_test Set5+Set14+B100+Urban100(or DIV2K) \
--scale 2+3+4+6+8 \
--n_GPUs 1 \
--save 'xxx/xxx' \
--crop_batch_size 1 \
--num_heads 6 \
--num_layers 6 
```
Here we release the checkpoint of 'HAT+ARIS' with 6 heads and 6 layers, more checkpoints will comming soon.

## Citation
If you use this code for your research or project, please cite:

	@article{zhou2022aris,
    author    = {Zhou, Qinye and Li, Ziyi and Xie, Weidi and Zhang, Xiaoyun and Zhang, Ya and Wang, Yanfeng},
    title     = {A Simple Plugin for Transforming Images to Arbitrary Scales},
    booktitle = {British Machine Vision Conference (BMVC)}，
    year      = {2022}
    }
	
## Acknowledgements
Many thanks to the code bases from [IPT](https://github.com/huawei-noah/Pretrained-IPT), [ArbSR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR).
