import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from typing import Tuple, Dict, Optional, List
from torchvision import transforms
from skimage import io

class RadioPointCloudDataset(Dataset):
    """
    Dataset for radio point cloud data used with Sionna RT
    Structure:
    - dataset/tx_train_positions.json and tx_test_positions.json: transmitter positions
    - dataset/gain/: target images named like "mapnumber_transmitternumber.png"
    - dataset/point_cloud/: point clouds stored as "mapnumber.txt"
    """
    
    def __init__(self,
                 phase: str = "train",
                 dataset_dir: str = "dataset/",
                 number_of_maps: int = 8,
                 numTx_train: int = 80,
                 numTx_test: int = 20,
                 transform=None):
        """
        Args:
            phase: "train", "val", "test"
            dataset_dir: directory of the dataset
            number_of_maps: Total number of maps available (maps 0 to number_of_maps-1)
            numTx_train: Number of transmitters for training/validation per map
            numTx_test: Number of transmitters for testing per map
            transform: Transform to apply
        """
        
        self.phase = phase
        self.number_of_maps = number_of_maps
        self.numTx_train = numTx_train
        self.numTx_test = numTx_test
        self.transform = transform
        
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # If dataset_dir is relative, build it relative to the script
        if not os.path.isabs(dataset_dir):
            self.dataset_dir = os.path.join(script_dir, dataset_dir)
        else:
            self.dataset_dir = dataset_dir

        
        self.tx_start = 0
        if phase == "train":
            self.tx_end = numTx_train
            tx_file = os.path.join(self.dataset_dir, "tx_train_positions.json")
        elif phase == "test":
            self.tx_end = numTx_test
            tx_file = os.path.join(self.dataset_dir, "tx_test_positions.json")
        
        self.transmitter_positions = self.load_transmitter_positions(tx_file)
        self.point_cloud_dir = os.path.join(self.dataset_dir, "point_cloud")
        self.dir_gain = os.path.join(self.dataset_dir, "gain")
        

    def __len__(self):
        return self.number_of_maps * self.tx_end
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        map_idx = idx // self.tx_end
        tx_idx_local = idx % self.tx_end

        actual_map_idx = map_idx
        actual_tx_idx = tx_idx_local
        sample_id = f"{actual_map_idx}_{actual_tx_idx}"

        # Create sample_id like the original format
        if self.phase == "train":
            image_id = sample_id
        else:  # test
            image_id = f"{actual_map_idx}_test_{actual_tx_idx}"

        point_cloud = self.load_point_cloud(str(actual_map_idx))
        tx_pos = self.transmitter_positions[sample_id]
        target_image = self.load_target_image(image_id)
        
        return (
            torch.from_numpy(point_cloud).float(),
            torch.from_numpy(tx_pos).float(),
            torch.from_numpy(target_image).float()
        )

    def load_transmitter_positions(self, transmitter_file: str) -> Dict:
        """
        Load transmitter positions from a JSON file.
        Expected format: list of lists of positions [[x1, y1, z1], [x2, y2, z2], ...]
        """
        with open(transmitter_file, "r") as f:
            data = json.load(f)

        transmitters = {}
        
        for map_idx, positions_list in enumerate(data):
            for tx_idx, position in enumerate(positions_list):
                key = f"{map_idx}_{tx_idx}"
                transmitters[key] = np.array(position, dtype=np.float32)
        
        return transmitters
    
    def load_point_cloud(self, map_number: str) -> np.ndarray:
        """
        Load and preprocess the point cloud from a .txt file.
        File format: x y z per line
        Returns: [N, 3] array with [x, y, z] for each point
        """
        file_path = os.path.join(self.point_cloud_dir, f"{map_number}.txt")
    
        points = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Ignore empty lines
                    coords = line.split()
                    if len(coords) >= 3:
                        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                        points.append([x, y, z])
        
        points = np.array(points, dtype=np.float32)
        
        return points
    
    def load_target_image(self, sample_id: str) -> np.ndarray:
        img_name_gain = os.path.join(self.dir_gain, sample_id + ".png")  
        image_gain = np.asarray(io.imread(img_name_gain)) / 255
        image_gain = np.expand_dims(image_gain, axis=0)  # (1, H, W)
        return image_gain
