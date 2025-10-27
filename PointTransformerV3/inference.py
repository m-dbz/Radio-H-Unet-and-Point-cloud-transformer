import torch
from torch.utils.data import DataLoader
from model import create_model
from loader import RadioPointCloudDataset
from utils import custom_collate_fn
from training import TrainingConfig
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def load_model_checkpoint(model_path, device):
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model_size = config.model_size
    model = create_model(model_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Model ({model_size}) loaded!")
    print(f"   - Epoch: {checkpoint['epoch']}")
    print(f"   - Best Val Loss: {checkpoint['best_val_loss']:.6f}")
    for key, value in vars(config).items():
        print(f"   {key:20}: {value}")
    
    return model


def save_comparison_image(pred_np, target_np, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Prediction
    im1 = ax1.imshow(pred_np, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('Prediction', fontsize=10)
    ax1.axis('off')
    
    # Ground Truth
    im2 = ax2.imshow(target_np, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('Ground Truth', fontsize=10)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')  # Reduced DPI from 150 to 100
    plt.close()


def process_dataset(model, dataset, loader, output_dir, phase, device):
    print(f"\nProcessing dataset {phase}...")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    with torch.no_grad():
        for batch_idx, (point_clouds, tx_positions, targets) in enumerate(loader):
            batch_size = tx_positions.size(0)
            
            tx_pos = tx_positions.to(device, non_blocking=True)
            target = targets.to(device, non_blocking=True)

            predictions = model(point_clouds, tx_pos)
            batch_predictions = []
            batch_targets = []
            batch_save_paths = []
            
            for i in range(batch_size):
                sample_idx = batch_idx * loader.batch_size + i
                map_idx = sample_idx // dataset.tx_end
                tx_idx = sample_idx % dataset.tx_end
                
                pred_np = predictions[i].squeeze().cpu().numpy()
                target_np = target[i].squeeze().cpu().numpy()
                
                filename = f"{map_idx}_{tx_idx}.png"
                save_path = os.path.join(output_dir, filename)
                save_comparison_image(pred_np, target_np, save_path)
                batch_predictions.append(pred_np)
                batch_targets.append(target_np)
                batch_save_paths.append(save_path)
                processed_count += 1
    
    print(f"   {processed_count} images generated")


def simple_inference():
    """Simple inference: generate predictions and save side-by-side comparisons"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MODEL INFERENCE")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(current_dir, "Examples")
    model_path = os.path.join(current_dir, 'best_model_default.pt')
    
    os.makedirs(examples_dir, exist_ok=True)
    print(f"📁 Output directory: {examples_dir}")

    model = load_model_checkpoint(model_path, device)
    
    for phase in ["train", "test"]:
        print(f"\nLoading dataset {phase}...")
        
        dataset = RadioPointCloudDataset(
            phase=phase, 
            number_of_maps=5,
            numTx_train=4,
            numTx_test=4
        )
        loader = DataLoader(
            dataset, 
            batch_size=4,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"   {len(dataset)} samples available")
        output_dir = os.path.join(examples_dir, phase)
        process_dataset(model, dataset, loader, output_dir, phase, device)
    
    print(f"\nINFERENCE COMPLETE!")
    print(f"Results available at: {examples_dir}")
    
    return examples_dir


if __name__ == "__main__":
    simple_inference()
