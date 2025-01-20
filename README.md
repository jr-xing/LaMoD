# LaMoD: Latent Motion Diffusion Model For Myocardial Strain Generation

## Introduction

Official Pytorch implementation for paper [LaMoD: Latent Motion Diffusion Model For Myocardial Strain Generation](https://arxiv.org/abs/2407.02229)

## Inference
Please refer to `inference.py` to see how to predict the displacement field and generating corresponding circumfenertial strain. The trained checkpoint files can be found [at here](https://drive.google.com/drive/folders/1Z99vlrvGoGVA-BQyIWQyDo2Fyw5HVkZ-?usp=drive_link) 

## Training

### Configuration

Before running the script, you need to set up a configuration file (e.g., `ours-LaMoD.json`). This file contains many sections, and the most important ones are:

- `data_split`: Specifies how to split the dataset into training, validation, and test sets.
- `datasets`: Contains paths and parameters for the datasets.
- `networks`: Defines the model architectures and parameters.

### Start Training

To start the training process, run the following command in your terminal:

```bash
python train.py --config configs/ours-LaMoD.json
```
