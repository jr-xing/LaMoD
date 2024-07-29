import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from modules.data import check_dict

# %%
# USE GPU 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # 1. Load Data

# %%
test_fname = '/scratch/jx8fh/2024-06-22-MICCAI-ShapeMI-Workshop-data/lamod_test.npy'
test_data = np.load(test_fname, allow_pickle=True).tolist()

# %%
check_dict(test_data[0])

# %% [markdown]
# # 2. Load Config

# %%
import json
config_fname = 'configs/ours-LaMoD.json'
config = json.load(open(config_fname))

# %% [markdown]
# # 3. Build Model

# %%
from models.LaMoD import LaMoD
network = LaMoD(config['networks'], device).to(device)

# %% [markdown]
# # 4. Inference

# %% [markdown]
# ## 4.1 Displacement Field

# %%
from skimage.transform import resize
inference_mask = torch.from_numpy(
    resize(test_data[check_data_idx]['myo_masks'][0,0]>0.5, [48,48], anti_aliasing=False)
)

# %%
check_data_idx = 0
check_frame_idx = 20

# Prepare Pytorch tensor
inference_input_video_data      = torch.from_numpy(test_data[check_data_idx]['myo_masks']).to(device)
inference_input_video_ori_frame = test_data[check_data_idx]['ori_n_frames']
inference_output_dict = network.inference(
    inference_input_video, 
    ori_n_frames=[inference_input_video_ori_frame])[0]

# Extract prediction
inference_disp = inference_output_dict['LaMoD_disp'].detach().cpu()

# Visualize
fig, axe = plt.subplots(1, 1, figsize=(5,5))
axe.quiver(
    inference_disp[0,0,check_frame_idx]*inference_mask,
    inference_disp[0,1,check_frame_idx]*inference_mask,
    units='xy',
    scale=1
)

# %% [markdown]
# ## 4.2 Strain Image

# %%
from modules.data.processing.strain_analysis.pixelstrain import pixelstrain

H, W = inference_disp.shape[-2:]
X, Y = np.meshgrid(np.arange(W), np.arange(H))
dXt = np.moveaxis(inference_disp[0,0].numpy(), 0, -1)
dYt = np.moveaxis(inference_disp[0,1].numpy(), 0, -1)
mask = inference_mask.numpy()
Nfr = inference_disp.shape[2]
strain = pixelstrain(X=X, Y=Y, dXt=dXt, dYt=dYt, mask=mask, times=np.arange(Nfr))
plt.imshow(strain['CC'][...,check_frame_idx], cmap='jet', vmin=-0.2, vmax=0.2)


