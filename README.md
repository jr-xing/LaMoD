# LaMoD: Latent Motion Diffusion Model For Myocardial Strain Generation

## Introduction

This repository contains the official PyTorch implementation of the paper **"LaMoD: Latent Motion Diffusion Model For Myocardial Strain Generation"**, which introduces a novel method for predicting highly accurate motion fields from standard cardiac magnetic resonance (CMR) imaging videos. The proposed Latent Motion Diffusion model (LaMoD) leverages a pre-trained registration network to extract latent motion features and employs a probabilistic latent diffusion model to reconstruct accurate motion fields, supervised by ground-truth motion data from displacement encoding with stimulated echoes (DENSE) CMR.

The paper is available on [arXiv](https://arxiv.org/abs/2407.02229).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jr-xing/LaMoD.git
   cd LaMoD
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure that you have a compatible version of PyTorch installed, preferably with GPU support.

3. **Download the pre-trained model weights**:
   The trained checkpoint files can be found [here](https://drive.google.com/drive/folders/1Z99vlrvGoGVA-BQyIWQyDo2Fyw5HVkZ-?usp=drive_link). Download the weights and place them in the appropriate directory.

## Training

### Configuration

Before running the training script, you need to set up a configuration file (e.g., `ours-LaMoD.json`). The configuration file contains several sections, including:

- **`data_split`**: Specifies how to split the dataset into training, validation, and test sets.
- **`datasets`**: Contains paths and parameters for the datasets.
- **`networks`**: Defines the model architectures and parameters.

### Start Training

To start the training process, run the following command:

```bash
python train.py --config configs/ours-LaMoD.json
```

The training script will automatically handle data loading, model training, and evaluation. You can monitor the training process using tools like [Weights & Biases (wandb)](https://wandb.ai/) if enabled in the configuration.

## Inference

### Predicting Displacement Fields and Strain

To predict displacement fields and generate corresponding circumferential strain, use the `inference.ipynb` script. The script takes input data in the form of a PyTorch tensor with shape `[1, n_frames, H, W]`, where `n_frames` is the number of sequence frames, and `H` and `W` are the frame height and width, respectively.

The script will output the predicted displacement fields and strain images, which can be visualized using matplotlib.

### Example Inference Workflow

1. **Load Data**: Ensure your input data is in the correct format (PyTorch tensor with shape `[1, n_frames, H, W]`).
2. **Load Config**: Load the configuration file that defines the model parameters.
3. **Build Model**: Initialize the LaMoD model using the provided configuration.
4. **Run Inference**: Predict the displacement fields and strain using the `network.inference` method.

## Results

The trained LaMoD model significantly improves the accuracy of motion analysis in standard CMR images, leading to better myocardial strain analysis in clinical settings for cardiac patients. Below are some example results from the paper:

- **Displacement Fields**: Visual comparison between predicted displacement fields and DENSE ground truth.
- **Strain Images**: Comparison of predicted circumferential strain with ground truth strain.

## Citation

If you use this code or the LaMoD model in your research, please cite the following paper:

```bibtex
@inproceedings{xing2024lamod,
  title={LaMoD: Latent Motion Diffusion Model for Myocardial Strain Generation},
  author={Xing, Jiarui and Jayakumar, Nivetha and Wu, Nian and Wang, Yu and Epstein, Frederick H and Zhang, Miaomiao},
  booktitle={International Workshop on Shape in Medical Imaging},
  pages={164--177},
  year={2024},
  organization={Springer}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on GitHub or contact the authors directly.

---

**Disclaimer**: This implementation is provided as-is, and the authors are not responsible for any errors or omissions. Use at your own risk.
