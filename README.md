# Few-Shot AI-Generated Image Detector

Official Pytorch implementation of paper:

> [Few-Shot Learner Generalizes Across AI-Generated Image Detection](https://arxiv.org/abs/2501.08763)
>
> Shiyu Wu, Jing Liu, Jing Li, Yequan Wang

Novel AI-generated image detector which is able to effectively distinguish unseen fake images by utilizing very few new samples. 

## Requirements

You can setup the environment as follows:

```python
# create conda environment
conda create -n FSD -y python=3.12
conda activate FSD

# install dependencies
pip install -r requirements.txt
```

## Getting Data & Directory structure

To download [GenImage](https://arxiv.org/abs/2306.08571) dataset, please refer to [this repository](https://github.com/GenImage-Dataset/GenImage) or download from [Baidu Yunpan](https://pan.baidu.com/share/init?surl=i0OFqYN5i6oFAxeK6bIwRQ) with code ztf1. 

<details>
<summary> Please organize the above data as follows: </summary>

```
data/
|-- GenImage/test
|   |-- ADM
|   |   |--train
|   |   |   |--0_adm_0.PNG
|   |   |   |......
|   |   |--val
|   |-- BigGAN
|   |-- glide
|   |-- Midjourney
|   |-- SD
|   |-- VQDM
|   |-- real
```

Real data are those nature images from stable_diffusion_v_1_4 and stable_diffusion_v_1_5. 
</details>


## Training

```
bash scripts/train.sh
```

This script enables training with 4 GPUs, you can specify the number of GPUs by setting `GPU_NUM`.

## Inference

```
bash scripts/eval.sh
```

Please specify the checkpoint directroy in the script. 

## Checkpoints
We provide our checkpoints trained on each test part for our cross-generator evaluation at [Baidu Yunpan](https://pan.baidu.com/s/1zNxDKtFJ_5KXcMceNtrRqA?pwd=icml) with code icml. 

## Citing
If you find this repository useful for your work, please consider citing it as follows:
```
@article{wu2025fsd,
  title={Few-Shot Learner Generalizes Across AI-Generated Image Detection},
  author={Shiyu Wu and Jing Liu and Jing Li and Yequan Wang},
  eprint={2501.08763},
  year={2025},
  journal={arXiv preprint arXiv:2501.08763},
  url={https://arxiv.org/abs/2501.08763}
}
```