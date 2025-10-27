import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import sys
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm
from model import create_model
from loader import RadioPointCloudDataset
from utils import custom_collate_fn


def clear_gpu_memory():
    """Optimized GPU memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TrainingConfig:
    """Training configuration"""
    
    def __init__(self):
        self.batch_size = 16 
        self.num_epochs = 400
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        
        self.number_of_maps = 50
        self.numTx_train = 80
        self.numTx_test = 20
        
        self.model_size = "default"
        
        self.save_frequency = 10
        
        self.scheduler_step_size = 10
        self.scheduler_gamma = 0.8

        self.use_tqdm = True
        self.tqdm_mininterval = 5.0


class Trainer:
    """Training class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.create_model()
        self.create_datasets()
        self.create_optimizer()
        
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.train_history = []
        self.val_history = []
        
        clear_gpu_memory()
        
        # Decide if tqdm should be disabled (e.g., on non-interactive Slurm jobs)
        self.disable_tqdm = (
            (not self.config.use_tqdm) or
            ('SLURM_JOB_ID' in os.environ and not sys.stderr.isatty()) or
            (os.environ.get('TQDM_DISABLE', '').strip() == '1')
        )
        
    def create_model(self):
        print(f"🤖 Creating model ({self.config.model_size})...")
        
        self.model = create_model(self.config.model_size).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Approximate size: {total_params * 4 / 1024 / 1024:.1f} MB")

    def create_datasets(self):
        print("Creating datasets...")
        
        self.train_dataset = RadioPointCloudDataset(
            phase="train",
            number_of_maps=self.config.number_of_maps,
            numTx_train=self.config.numTx_train,
            numTx_test=self.config.numTx_test
        )
        
        self.val_dataset = RadioPointCloudDataset(
            phase="test",
            number_of_maps=self.config.number_of_maps,
            numTx_train=self.config.numTx_train,
            numTx_test=self.config.numTx_test
        )
        
        print(f"   Train: {len(self.train_dataset)} samples")
        print(f"   Val: {len(self.val_dataset)} samples")
        
        self.create_dataloaders()
    
    
    def create_dataloaders(self):
        collate_fn = custom_collate_fn
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
    
    def create_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )
        
        print(f"   Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"   Scheduler: StepLR (step={self.config.scheduler_step_size}, gamma={self.config.scheduler_gamma})")
    
    def train_epoch(self, epoch):
        """Train one epoch with error handling"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        num_errors = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch+1}/{self.config.num_epochs}",
            leave=False,
            disable=self.disable_tqdm,
            mininterval=self.config.tqdm_mininterval
        )
        
        for batch_idx, (point_clouds, tx_positions, targets) in enumerate(pbar):
            try:
                tx_positions = tx_positions.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                outputs = self.model(point_clouds, tx_positions)

                loss = self.criterion(outputs, targets)
                loss.backward()
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'avg_loss': f"{total_loss/num_batches:.6f}",
                    'lr': f"{current_lr:.2e}",
                    'errors': num_errors
                })
                
            except Exception as e:
                num_errors += 1
                print(f"Error in batch {batch_idx}: {e}")
                clear_gpu_memory()
                continue
        
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'loss': avg_loss,
            'lr': current_lr,
            'num_batches': num_batches,
            'num_errors': num_errors
        }
    
    def validate_epoch(self):
        """Validate one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        num_errors = 0
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="Validation",
                leave=False,
                disable=self.disable_tqdm,
                mininterval=self.config.tqdm_mininterval
            )
            
            for batch_idx, (point_clouds, tx_positions, targets) in enumerate(pbar):
                try:
                    tx_positions = tx_positions.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    outputs = self.model(point_clouds, tx_positions)
                    
                    loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({
                        'val_loss': f"{loss.item():.6f}",
                        'avg_val_loss': f"{total_loss/num_batches:.6f}"
                    })
                    
                except Exception as e:
                    num_errors += 1
                    if num_errors <= 3:  # Show only the first few errors
                        print(f"Validation error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches,
            'num_errors': num_errors
        }
    
    def train(self, resume_from_checkpoint=None):
        print("Start training")
        print("=" * 80)
        
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                epoch_start = time.time()

                print(f"\nEPOCH {epoch+1}/{self.config.num_epochs}")
                print("-" * 50)
                
                train_results = self.train_epoch(epoch)
                val_results = self.validate_epoch()
                
                epoch_time = time.time() - epoch_start
                
                # Save best model
                is_best = False
                previous_best = self.best_val_loss
                if val_results['loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['loss']
                    is_best = True
                    self.save_model(epoch, is_best=True)
                
                # Display results
                print(f"\nEPOCH {epoch+1} RESULTS:")
                print(f"   Time: {epoch_time:.1f}s")
                print(f"   Train - Loss: {train_results['loss']:.6f} (batches: {train_results['num_batches']}, errors: {train_results['num_errors']})")
                print(f"   Val   - Loss: {val_results['loss']:.6f} (batches: {val_results['num_batches']}, errors: {val_results['num_errors']})")
                print(f"   LR: {train_results['lr']:.2e}")

                if is_best:
                    print(f"   ⭐ NEW BEST! (Previous: {previous_best:.6f})")
                
                self.train_history.append(train_results)
                self.val_history.append(val_results)
                
                if epoch % self.config.save_frequency == 0:
                    self.save_model(epoch)
                
                clear_gpu_memory()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        except Exception as e:
            print(f"\nCritical error during training: {e}")
            traceback.print_exc()
        
        finally:
            self.finalize_training()
    
    def save_model(self, epoch, is_best=False):
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'train_history': self.train_history,
                'val_history': self.val_history
            }
            
            if is_best:
                path = os.path.join(self.current_dir, f'best_model_{self.config.model_size}.pt')
                torch.save(checkpoint, path)
                print(f"Best model saved: {os.path.basename(path)}")
            else:
                path = os.path.join(self.current_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save(checkpoint, path)
                print(f"Checkpoint saved: {os.path.basename(path)}")
                
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")
    
    def load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
            
        print(f"Checkpoint loaded: resuming at epoch {self.start_epoch}")
    
    def finalize_training(self):
        print(f"\nTRAINING COMPLETE!")
        print("=" * 50)
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        print(f"   Epochs trained: {len(self.train_history)}")
        
        final_path = os.path.join(self.current_dir, f'final_model_{self.config.model_size}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, final_path)

        print(f"   Final model saved: {os.path.basename(final_path)}")
        
        self.plot_training_curves()
        clear_gpu_memory()
    
    def plot_training_curves(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(self.train_history) + 1)
        
        train_losses = [h['loss'] for h in self.train_history]
        val_losses = [h['loss'] for h in self.val_history]
        
        ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
        ax.set_title('MSE Loss Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(self.current_dir, f'training_curves_{self.config.model_size}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   Curves saved: {os.path.basename(plot_path)}")
        

def main():
    try:
        config = TrainingConfig()
        
        print("CONFIGURATION:")
        print("-" * 30)
        for key, value in vars(config).items():
            print(f"   {key:20}: {value}")
        
        print(f"\nENVIRONMENT:")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Device name: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        print("\n" + "=" * 80)
        trainer = Trainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
