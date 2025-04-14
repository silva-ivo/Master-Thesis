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
import json



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
model_config_file = "/data/home/silva/Documents/Pipline_2/Results/UNET_GridSearch/Phase_1"
os.makedirs(model_config_file, exist_ok=True)

# === Model parameter space ===
dropout_rates = [0, 0.1, 0.3]

# === Training parameter space ===
window_size = {"5s": 1280, "10s": 2560, "30s": 7680, "1min": 15360}
batch_size = {"32": 32}
loss_function = {
    "RRMSELoss": ut.RRMSELoss(),
    "MSELoss": nn.MSELoss(),
    "MAELoss": nn.L1Loss(),
}
lr = {"0.01": 0.01, "0.001": 0.001, "0.0001": 0.0001}

# === Available models ===
models_dict = {
    "UNet_3": DAE_models.UNet_3,
    "UNet_4": DAE_models.UNet_4,
    "UNet_5": DAE_models.UNet_5,
}

model_id = 0

for model_name, model_class in models_dict.items():
    for dropout in dropout_rates:
        # === Model directory ===
        model_config_dir = os.path.join(model_config_file, f"{model_name}_dr{dropout}")
        os.makedirs(model_config_dir, exist_ok=True)

        # === Save model config as JSON ===
        model_config = {
            "model_name": model_name,
            "dropout_rate": dropout,
        }
        with open(os.path.join(model_config_dir, "config.json"), "w") as f:
            json.dump(model_config, f, indent=4)

        # === For each valid training config ===
        for win_size_key, batch_size_key, loss_fn_key, lr_key in itertools.product(
            window_size.keys(), batch_size.keys(), loss_function.keys(), lr.keys()
        ):
            # === Unpack parameters ===
            window_size_value = window_size[win_size_key]
            batch_size_value = batch_size[batch_size_key]
            loss_function_value = loss_function[loss_fn_key]
            lr_value = lr[lr_key]

            print(f"{model_name}_dr{dropout}")
            print(f"  Model: {model_name}")
            print(f"  Dropout: {dropout}")
            print(f"  Window: {win_size_key}, Batch: {batch_size_key}, Loss: {loss_fn_key}, LR: {lr_key}")

            # === Data ===
            train_loader, val_loader, test_loader = dataloader.get_dataloaders(
                data_base_dir, window_size_value, batch_size_value,
                split_ratio=(0.15, 0.045, 0.1)
            )

            # === Model ===
            model = model_class(
                input_channels=2,
                num_classes=2,
                dropout=dropout
            )

            device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

            # === Training ===
            model, history, x_input, y_true, y_pred = trainer.train_model(
                model, f"{model_name}_dr{dropout}", train_loader, val_loader, win_size_key,
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
                "model": model_name,
                "dropout": dropout,
                "window_size": win_size_key,
                "batch_size": batch_size_key,
                "loss_function": loss_fn_key,
                "learning_rate": lr_key,
                "pcc": pcc,
                "snr_diff": snr_diff,
                "rmse": rmse,
                "rrmse": rrmse
            }

            # Save to CSV

            result_df = pd.DataFrame([result])
            result_file = os.path.join(model_config_dir,  f"{model_name}_dr{dropout}_results.csv")
            result_df.to_csv(result_file, mode='a', index=False, header=not os.path.exists(result_file))

            # Optional: Save plots
            # ut.plot_loss(history, loss_fn_key, "GridSearch", f"model{model_id}", win_size_key)
            # ut.plot_predictions(y_true, y_pred, x_input, 10, loss_fn_key, "GridSearch", f"model{model_id}", win_size_key)

            