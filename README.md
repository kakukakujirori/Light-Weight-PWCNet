<div align="center">

# Light Weight PWCNet

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2207.11617)
[![Conference](http://img.shields.io/badge/SIGGRAPH-2022-4b44ce.svg)](https://www.wslai.net/publications/fusion_deblur/)

</div>

## Description

In the paper "Face Deblurring using Dual Camera Fusion on Mobile Phones", they use a light weight version of PWCNet. This repo aims to implement that part by myself.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/kakukakujirori/Light-Weight-PWCNet.git
cd Light-Weight-PWCNet

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Put as many natural images as you have in `data/train/` directory. I use [AutoFlow](https://autoflow-google.github.io/) to automatically generate synthetic data from them.

Also for evaluation, I temporally use Sitel dataset. [Download (5.3GB)](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip) and place the unzipped folder under `data/` with the name `Sintel/`.

NOTE: The accuracy is currently not satisfactory... Improvement under way.

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
