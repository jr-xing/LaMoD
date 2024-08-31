# LaMoD: Latent Motion Diffusion Model For Myocardial Strain Generation

## Introduction
Official Pytorch implementation for paper [LaMoD: Latent Motion Diffusion Model For Myocardial Strain Generation](https://arxiv.org/abs/2407.02229)
## How to Use `train.py`

### Configuration
Before running the script, you need to set up a configuration file (e.g., `config.yaml`). This file should include the following sections:
- `data_split`: Specifies how to split the dataset into training, validation, and test sets.
- `datasets`: Contains paths and parameters for the datasets.
- `networks`: Defines the model architectures and parameters.

### Training
To start the training process, run the following command in your terminal:

```bash
python train.py --config path/to/config.json
```

### Inference
Will be added soon.
