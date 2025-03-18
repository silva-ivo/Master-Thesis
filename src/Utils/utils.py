import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import os
import numpy as np

import matplotlib.pyplot as plt


def load_data(file_pattern,base_dir):
    file_list = sorted(glob.glob(os.path.join(base_dir, file_pattern)))
    if not file_list:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")
    data = [np.load(file, allow_pickle=True) for file in file_list]
    return np.array(data)  # Ensure output is a NumPy array

def split_segments(inputs, targets, window_size):
   
    num_segments, total_timepoints, num_channels, extra_dim = inputs.shape
    num_splits = total_timepoints // window_size  # Number of new windows per segment

    # Reshape inputs and targets into small windows
    split_inputs = inputs.reshape(num_segments, num_splits, window_size, num_channels, extra_dim)
    split_targets = targets.reshape(num_segments, num_splits, window_size, num_channels, extra_dim)

    # Remove the last dimension if it's redundant
    split_inputs = split_inputs.squeeze(-1)
    split_targets = split_targets.squeeze(-1)

    # Merge first two dimensions (all segments become one large batch)
    split_inputs = split_inputs.reshape(-1, window_size, num_channels)
    split_targets = split_targets.reshape(-1, window_size, num_channels)

    return split_inputs, split_targets

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Define custom loss function

class RRMSELoss(nn.Module):

    def __init__(self):
        super(RRMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))

        rms_y = torch.sqrt(torch.mean(y_true ** 2))+1e-8

        rrmse = torch.divide(rmse, rms_y)

        return rrmse
    
def plot_predictions(y_true_cpu, y_pred_cpu, x_inputs_cpu, 
                     num_windows, loss_function_name, mode, model_name, window_size_name, 
                     save_folder="/data/home/silva/Documents/Pipline_2/Results"):

    save_dir = os.path.join(save_folder, mode, model_name, loss_function_name, window_size_name)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_windows):
        random_index = torch.randint(0, len(y_true_cpu), (1,)).item()
        channel_idx = torch.randint(0, y_true_cpu.shape[2], (1,)).item()

        y_true_np = y_true_cpu[random_index, :, channel_idx].cpu().numpy()
        y_pred_np = y_pred_cpu[random_index, :, channel_idx].cpu().numpy()
        x_input_np = x_inputs_cpu[random_index, :, channel_idx].cpu().numpy()

        jitter = np.linspace(-0.2, 0.2, len(y_true_np))

        # ---- First Figure: Combined Plot ----
        plt.figure(figsize=(12, 6))
        plt.plot(x_input_np + jitter, label='Input Values', linestyle='-', linewidth=1.2, alpha=1, color='#D55E00')  # Dark orange
        plt.plot(y_true_np + jitter, label='True Values', linestyle='-', linewidth=1, alpha=0.8, color='#56B4E9')  # Light blue
        plt.plot(y_pred_np + jitter, label='Predicted Values', linestyle='-', linewidth=0.8, alpha=0.6, color='black')  # Black
        plt.title(f'Predictions vs True Values (Window {random_index}, Channel {channel_idx}, Loss: {loss_function_name})')
        plt.xlabel('Time Points')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

       
        save_path1 = os.path.join(save_dir, f'combined_plot_loss-{loss_function_name}_window-{window_size_name}_channel-{channel_idx}_{i+1}.png')
        plt.savefig(save_path1)
        plt.close()

        print(f"Plot saved at: {save_path1}")

        # # ---- Second Figure: Separate Plots ----
        # fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # # Get global min/max for consistent scaling
        # y_min = min(x_input_np.min(), y_true_np.min(), y_pred_np.min())
        # y_max = max(x_input_np.max(), y_true_np.max(), y_pred_np.max())

        # # Plot Input Values
        # axes[0].plot(x_input_np + jitter, linestyle='-', linewidth=1.2, alpha=1, color='#10b981')
        # axes[0].set_title("Input Values")
        # axes[0].grid(True)
        # axes[0].set_ylim(y_min, y_max)

        # # Plot True Values
        # axes[1].plot(y_true_np + jitter, linestyle='-', linewidth=1, alpha=0.8, color='#3b82f6')
        # axes[1].set_title("True Values")
        # axes[1].grid(True)
        # axes[1].set_ylim(y_min, y_max)

        # # Plot Predicted Values
        # axes[2].plot(y_pred_np + jitter, linestyle='-', linewidth=0.8, alpha=0.6, color='#d97706')
        # axes[2].set_title("Predicted Values")
        # axes[2].grid(True)
        # axes[2].set_ylim(y_min, y_max)

        # plt.xlabel("Time Points")
        # plt.tight_layout()

        # # Save the second figure
        # save_path2 = os.path.join(save_folder, f'separated_plots_loss-{loss_function}_window-{window_label}_channel-{channel_idx}_{i+1}.png')
        # plt.savefig(save_path2)
        # plt.close()
     

def plot_loss(history, loss_function_name, mode, model_name, window_size_name, 
              save_folder="/data/home/silva/Documents/Pipline_2/Results"):
    

    save_dir = os.path.join(save_folder, mode, model_name, loss_function_name, window_size_name)
    os.makedirs(save_dir, exist_ok=True)

    
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', linestyle='-', marker='o', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='-', marker='s', color='red')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

   
    save_path = os.path.join(save_dir, 'History_loss.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Loss plot saved at: {save_path}")