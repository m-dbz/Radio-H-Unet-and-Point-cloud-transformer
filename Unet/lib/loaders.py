import os
import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

class RadioUNet_c(Dataset):
    """
    Loader:
      inputs: tensor (C=2, H=256, W=256)  (heightmap, tx_map)
      target: tensor (1, H, W) at the requested height
      height_scalar: tensor (1,) normalized in [0,1] (h / max_height)
    Filenames expected:
      train/val: antennas/{map}_{tx}_{height}.png
      test:     antennas/{map}_{tx}_{height}_test.png
    """
    def __init__(self, phase="train",
                 dir_dataset="content/",
                 number_of_maps=2,
                 numTx_train=80,
                 numTx_test=40,
                 heights=[1,2,4,8],
                 thresh=0.2,
                 transform=transforms.ToTensor()):
        self.phase = phase
        self.dir_dataset = dir_dataset
        self.thresh = thresh
        self.transform = transform
        self.number_of_maps = number_of_maps
        self.numTx_train = numTx_train
        self.numTx_test = numTx_test

        self.heights_list = list(heights)
        self.max_height = float(max(self.heights_list))

        if phase == "train":
            self.numTx = numTx_train
        else:
            self.numTx = numTx_test

        self.dir_gain = os.path.join(self.dir_dataset, "gain")
        self.dir_buildings = os.path.join(self.dir_dataset, "buildings_complete")
        self.dir_Tx = os.path.join(self.dir_dataset, "antennas")

        self.height = 256
        self.width = 256

    def __len__(self):
        return self.number_of_maps * self.numTx * len(self.heights_list)

    def __getitem__(self, idx):
        n_heights = len(self.heights_list)
        num_tx = self.numTx
        actual_map_idx = idx // (num_tx * n_heights)
        rem = idx % (num_tx * n_heights)
        actual_tx_idx = rem // n_heights
        height_idx = rem % n_heights
        height_val = self.heights_list[height_idx]

        name_buildings = f"{actual_map_idx}.png"
        if self.phase in ["train", "val"]:
            name_tx = f"{actual_map_idx}_{actual_tx_idx}_{height_val}.png"
        else:
            name_tx = f"{actual_map_idx}_{actual_tx_idx}_{height_val}_test.png"

        img_build_path = os.path.join(self.dir_buildings, name_buildings)
        img_tx_path = os.path.join(self.dir_Tx, name_tx)
        img_gain_path = os.path.join(self.dir_gain, name_tx)

        image_buildings = np.asarray(io.imread(img_build_path)).astype(np.float32)
        image_Tx = np.asarray(io.imread(img_tx_path)).astype(np.float32)
        image_gain = np.expand_dims(np.asarray(io.imread(img_gain_path)).astype(np.float32), axis=2) / 255.0

        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones_like(image_gain)
            image_gain = image_gain / (1 - self.thresh)

        # stack inputs: (H,W,2)
        inputs = np.stack([image_buildings, image_Tx], axis=2)

        # transforms -> tensors (C,H,W)
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)

        h_norm = np.array([height_val / self.max_height], dtype=np.float32)
        h_tensor = torch.from_numpy(h_norm)

        return [inputs, image_gain, h_tensor]
