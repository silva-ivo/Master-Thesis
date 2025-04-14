from Utils import dataloader
from Utils import utils as ut
from Utils import metrics   
from models import cnn_models,DAE_models
from train import trainer
import torch
from torch import nn
import itertools
import pandas as pd
import os


#LOADERS
data_base_dir = "/data/home/silva/Documents/Pipline_2/Data"
# window_size = 15360
# window_size_name = "5s"
# batch_size = 64
#window_size = 15360 1minuto
#windows_size = 7680 30segundos
#windows_size = 2560 10segundos
#window_szie = 1280 5segundos
#window_size = 256 1segundos


# === Paths ===
model_config_file = "/data/home/silva/Documents/Pipline_2/Results/DCNN_GridSearch/Phase_1"
os.makedirs(os.path.dirname(model_config_file), exist_ok=True)

# === Model parameter space ===
num_blocks_options = [2, 3, 4]
channels_options = [[32, 64], [32, 64, 128], [32, 64, 128, 256]]
kernel_sizes_options = [[7, 5], [7, 5, 3], [9, 7, 5, 3]]
use_residual_options = [True, False]
dropout_rates = [0, 0.1, 0.3]

# === Training parameter space ===
window_size = {"5s": 1280, "10s": 2560, "30s": 7680, "1min": 15360}
batch_size = {"32": 32}
loss_function = {
    "RRMSELoss": ut.RRMSELoss(),
    "MSELoss": nn.MSELoss(),
    "MAELoss": nn.L1Loss(),
}
lr = {"0.01": 0.01,"0.001": 0.001, "0.0001": 0.0001}

model_id=0
  
    
for num_blocks in num_blocks_options:
    for channels in channels_options:
        for kernels in kernel_sizes_options:
            # Filter out invalid configs
            if len(channels) != num_blocks and len(kernels) != num_blocks:
                continue  # Skip invalid

            for use_res in use_residual_options:
                for dropout in dropout_rates:
                    # === Save model config ===
                    model_config = {
                        "model_id": model_id,
                        "num_blocks": num_blocks,
                        "channels": str(channels),
                        "kernel_sizes": str(kernels),
                        "use_residual": use_res,
                        "dropout_rate": dropout
                    }
                    df_model_config = pd.DataFrame([model_config])
                    os.makedirs(os.path.join(model_config_file, f"model{model_id}"), exist_ok=True)
                    model_config_file_id = os.path.join(model_config_file,f"model{model_id}", f"model{model_id}_config.csv")
                    print(f"Saving model config to {model_config_file_id}")
                    df_model_config.to_csv(model_config_file_id, mode='a', index=False,
                                            header=not os.path.exists(model_config_file_id))
                
                    
                    
                    # === For each valid training config ===
                    for win_size_key, batch_size_key, loss_fn_key, lr_key in itertools.product(
                        window_size.keys(), batch_size.keys(), loss_function.keys(), lr.keys()
                    ):
                        window_size_value = window_size[win_size_key]
                        batch_size_value = batch_size[batch_size_key]
                        loss_function_value = loss_function[loss_fn_key]
                        lr_value = lr[lr_key]

                        print(f"\nModel {model_id}:")
                        print(f"  Blocks: {num_blocks}, Channels: {channels}, Kernels: {kernels}")
                        print(f"  Residual: {use_res}, Dropout: {dropout}")
                        print(f"  Window: {win_size_key}, Loss: {loss_fn_key}, LR: {lr_key}")

                        # === Get dataloaders ===
                        train_loader, val_loader, test_loader = dataloader.get_dataloaders(
                            data_base_dir, window_size_value, batch_size_value,
                            split_ratio=(0.15, 0.045, 0.1)
                        )

                        # === Model ===
                        model = cnn_models.EEGResNet1D(
                            input_channels=2,
                            num_blocks=num_blocks,
                            channels=channels,
                            kernel_sizes=kernels,
                            use_residual=use_res,
                            dropout_rate=dropout
                        )
                        model_name = f"model{model_id}"
                        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

                        # === Train ===
                        model, history, x_input, y_true, y_pred = trainer.train_model(
                            model, model_name, train_loader, val_loader, win_size_key,
                            loss_function_value, loss_fn_key, device, num_epochs=400,
                            early_stopping_patience=20, lr=lr_value
                        )

                        # === Metrics ===
                        pcc = metrics.compute_pcc(y_true, y_pred)
                        snr_diff = metrics.compute_snr_diff(y_true, y_pred, x_input)
                        rmse = metrics.compute_rmse(y_true, y_pred)
                        rrmse = metrics.compute_rrmse(y_true, y_pred)

                        # === Save training result ===
                        result = {
                            "model_id": model_id,
                            "window_size": win_size_key,
                            "batch_size": batch_size_key,
                            "loss_function": loss_fn_key,
                            "learning_rate": lr_key,
                            "pcc": pcc,
                            "snr_diff": snr_diff,
                            "rmse": rmse,
                            "rrmse": rrmse
                        }
                        os.makedirs(os.path.join(model_config_file, f"model{model_id}"), exist_ok=True)
                        df_result = pd.DataFrame([result])
                        model_results_file_id = os.path.join(model_config_file,f"model{model_id}", f"model{model_id}_results.csv")
                        df_result.to_csv(model_results_file_id, mode='a', index=False,
                                         header=not os.path.exists(model_results_file_id))

                        # # === Plots (optional) ===
                        # ut.plot_loss(history, loss_fn_key, "GridSearch", model_name, win_size_key)
                        # ut.plot_predictions(y_true, y_pred, x_input, 10, loss_fn_key, "GridSearch", model_name, win_size_key)

                    model_id += 1