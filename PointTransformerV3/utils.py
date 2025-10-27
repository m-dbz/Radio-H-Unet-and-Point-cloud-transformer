import torch

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized point clouds"""
    point_clouds = []
    tx_positions = []
    targets = []
    
    for pc, tx, target in batch:
        if pc.shape[0] > 0:
            point_clouds.append(pc)
            tx_positions.append(tx)
            targets.append(target)
    
    if len(point_clouds) == 0:
        return [], torch.empty(0, 3), torch.empty(0, 1, 256, 256)
    
    tx_positions = torch.stack(tx_positions, dim=0)
    targets = torch.stack(targets, dim=0)
    
    return point_clouds, tx_positions, targets
