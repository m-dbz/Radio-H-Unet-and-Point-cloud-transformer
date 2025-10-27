
from __future__ import print_function, division
import os
from cv2 import phase
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


from lib import loaders, modules


numTx_train = 80
numTx_test = 20
number_of_maps = 500
heights = [1, 2, 4, 8, 16, 32, 64, 128]

Radio_train = loaders.RadioUNet_c(phase="train", numTx_train=numTx_train,
                                 numTx_test=numTx_test, number_of_maps=number_of_maps,
                                 heights=heights)
Radio_val = loaders.RadioUNet_c(phase="test", numTx_train=numTx_train,
                               numTx_test=numTx_test, number_of_maps=number_of_maps,
                               heights=heights)
Radio_test = loaders.RadioUNet_c(phase="test", numTx_train=numTx_train,
                                numTx_test=numTx_test, number_of_maps=number_of_maps,
                                heights=heights)

print(f"Train samples: {len(Radio_train)}")
print(f"Val samples: {len(Radio_val)}")
print(f"Test samples: {len(Radio_test)}")

image_datasets = {
    'train': Radio_train, 'val': Radio_val
}

batch_size = 16

dataloaders = {
    'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1),
    'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled = True

model = modules.RadioWNet(phase="firstU")
model.cuda()

def plot_training_losses(train_losses, val_losses, log_scale=False, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss' + (' (log scale)' if log_scale else ''), fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    else:
        plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Total epochs: {len(epochs)}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best validation loss: {min(val_losses):.6f} at epoch {epochs[np.argmin(val_losses)]}")
    print(f"Best training loss: {min(train_losses):.6f} at epoch {epochs[np.argmin(train_losses)]}")

def calc_loss_dense(pred, target, metrics):
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def calc_loss_sparse(pred, target, samples, metrics, num_samples):
    criterion = nn.MSELoss()
    loss = criterion(samples*pred, samples*target)*(256**2)/num_samples
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs1)))

def train_model(model, optimizer, scheduler, num_epochs=50, WNetPhase="firstU", num_samples=50):
    # WNetPhase: traine first U and freez second ("firstU"), or vice verse ("secondU").
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("learning rate", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, targets, heights in dataloaders[phase]:
                inputs = inputs.to(device)        # (B, C=2, H, W)
                targets = targets.to(device)      # (B, 1, H, W)
                heights = heights.to(device)      # (B, 1) normalized

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, heights)   # returns [out1, out2]
                    outputs1, outputs2 = outputs
                    if WNetPhase == "firstU":
                        loss = calc_loss_dense(outputs1, targets, metrics)
                    else:
                        loss = calc_loss_dense(outputs2, targets, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)


            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)


            if epoch_loss < best_loss: #phase == 'val' and:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

 
# Training First UNet
optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.90)
model, train_losses, val_losses = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs = 50)

plot_training_losses(train_losses, val_losses, log_scale=True, save_path='Training_Losses_FirstU.png')

torch.save(model.state_dict(), 'Trained_Model_FirstU.pt')

model = modules.RadioWNet(phase="firstU")
model.load_state_dict(torch.load('Trained_Model_FirstU.pt'))
model.to(device)

 
# Second U Module
model = modules.RadioWNet(phase="secondU")
model.load_state_dict(torch.load('Trained_Model_FirstU.pt'))
model.to(device)

 
# Training Second UNet

optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.90)
model, train_losses_second, val_losses_second = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=50, WNetPhase="secondU")

plot_training_losses(train_losses_second, val_losses_second, log_scale=True, save_path='Training_Losses_SecondU.png')

# Save Second U Model For Inference
torch.save(model.state_dict(), 'Trained_Model_SecondU.pt')

model = modules.RadioWNet(phase="secondU")
model.load_state_dict(torch.load('Trained_Model_SecondU.pt'))
model.to(device)

# Test Accuracy
def calc_loss_test(pred1, pred2, target, metrics, error="MSE"):
    criterion = nn.MSELoss()
    if error=="MSE":
        loss1 = criterion(pred1, target)
        loss2 = criterion(pred2, target)
    else:
        loss1 = criterion(pred1, target)/criterion(target, 0*target)
        loss2 = criterion(pred2, target)/criterion(target, 0*target)
    metrics['loss first U'] += loss1.data.cpu().numpy() * target.size(0)
    metrics['loss second U'] += loss2.data.cpu().numpy() * target.size(0)

    return [loss1,loss2]

def print_metrics_test(metrics, epoch_samples, error):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format("Test"+" "+error, ", ".join(outputs)))

def test_loss(model, error="MSE"):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs, targets, heights in DataLoader(Radio_test, batch_size=batch_size, shuffle=True, num_workers=1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        heights = heights.to(device)
        with torch.set_grad_enabled(False):
            outputs1, outputs2 = model(inputs, heights)
            [loss1, loss2] = calc_loss_test(outputs1, outputs2, targets, metrics, error)
            epoch_samples += inputs.size(0)

    print_metrics_test(metrics, epoch_samples, error)
    #test_loss1 = metrics['loss U'] / epoch_samples
    #test_loss2 = metrics['loss W'] / epoch_samples
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# MSE NMSE Accuracy on DPM

test_loss(model,error="MSE")
test_loss(model,error="NMSE")


# -----------------------------------------------------------------------------
# Side-by-side prediction vs ground truth saving (inference-style)
# -----------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # avoid GUI issues

def save_comparison_image(pred_np, target_np, save_path):
    """Save a side-by-side figure of prediction and ground truth (arrays in [0,1])."""
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(pred_np, cmap='viridis', vmin=0, vmax=1)
    plt.title('Prediction', fontsize=8)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(target_np, cmap='viridis', vmin=0, vmax=1)
    plt.title('Ground Truth', fontsize=8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()

def save_dataset_examples(model, dataset, tx_per_map, phase_name, base_dir):
    out_dir = os.path.join(base_dir, phase_name)
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    count = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            inp = sample[0].unsqueeze(0).to(device)         # (1, C, H, W)
            tgt = sample[1].unsqueeze(0).to(device)         # (1, 1, H, W)
            h = sample[2].unsqueeze(0).to(device)           # (1, 1)
            map_id = idx // (tx_per_map * len(dataset.heights_list))
            tx_idx = (idx // len(dataset.heights_list)) % tx_per_map
            height_val = dataset.heights_list[idx % len(dataset.heights_list)]
            out1, out2 = model(inp, h)
            pred = out2 if out2 is not None else out1
            pred_np = pred[0,0].clamp(0,1).cpu().numpy()
            tgt_np = tgt[0,0].clamp(0,1).cpu().numpy()
            file_name = f"{map_id}_{tx_idx}_{height_val}.png"
            save_path = os.path.join(out_dir, file_name)
            save_comparison_image(pred_np, tgt_np, save_path)
            count += 1
    return count


# Small focused datasets (first 5 maps, 4 tx each) provided by new variables
Radio_train2 = loaders.RadioUNet_c(phase="train", numTx_train=4, numTx_test=4, number_of_maps=5)
Radio_val2 = loaders.RadioUNet_c(phase="test", numTx_train=4, numTx_test=4, number_of_maps=5)

print('\n=== Generating side-by-side examples (RadioWNet small sets) ===')
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_root = os.path.join(current_dir, 'Examples')
os.makedirs(examples_root, exist_ok=True)
saved_train2 = save_dataset_examples(model, Radio_train2, 4, 'train_small', examples_root)
saved_test2 = save_dataset_examples(model, Radio_val2, 4, 'test_small', examples_root)
print(f"Small set examples saved: train_small={saved_train2}, test_small={saved_test2}")
print(f"All small example images written under: {examples_root}")
 