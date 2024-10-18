
# Bayesian Principles Improve Prompt Learning In VLMs

This repo contains the codebase of CLIP experiments.

## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated).

## Dataset Preparation

In this paper, we follow [DATASETS.md](DATASETS.md) to install the datasets. The task definition and few-shot learning setting are similar to following papers for fair comparison:
* [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557), in CVPR, 2022.
* [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134), in IJCV, 2022.

## How to Run

Click a paper below to see the detailed instructions on how to run the code to reproduce the results.

* [Bayesian Principles Improve Prompt Learning In VLMs](BP_CLIP.md)
