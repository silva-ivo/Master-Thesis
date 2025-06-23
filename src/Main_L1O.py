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
import numpy as np


data_base_dir = "/data/home/silva/Documents/Pipline_2/Data"

# === Paths ===

all_pcc,all_snr_diff,all_rmse,all_rrmse,all_cpt = [],[],[],[],[]

patient_data=dataloader.load_all_patients_grouped(data_base_dir, window_size=1280)
fold=1
for test_patient_id,train_loader,val_loader in dataloader.get_leave_one_patient_out_loaders(patient_data):
    
    print(f"Processing patient {test_patient_id}, fold-{fold}")
    model_id = f"{test_patient_id}_model"
    model_config_file = f"/data/home/silva/Documents/Pipline_2/Results/Final_Validation/{model_id}"
    os.makedirs(os.path.dirname(model_config_file), exist_ok=True)
    
    # === Model parameter space ===
    num_blocks = 4
    channels = [32, 64, 128, 256]
    kernel_sizes = [9, 7, 5, 3]
    use_residual = True
    dropout_rate = 0.1
    
    # === Training parameter space ===
    batch_size = 32
    loss_function = ut.RRMSELoss()
    lr = 0.001
    reduction = 4
    
    
    # Initialize model
    model = cnn_models.SE_ResNet1D(
        num_blocks=num_blocks,
        channels=channels,
        kernel_sizes=kernel_sizes,
        use_residual=use_residual,
        dropout_rate=dropout_rate,
        reduction=reduction
    )
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # === Train ===
    model, history, x_input, y_true, y_pred = trainer.train_model(
        model, model_id, train_loader, val_loader,loss_function, device, num_epochs=400,
        early_stopping_patience=20, lr=lr
    )
      # === Metrics ===
    pcc = metrics.compute_pcc(y_true, y_pred)
    snr_diff = metrics.compute_snr_diff(y_true, y_pred, x_input)
    rmse = metrics.compute_rmse(y_true, y_pred)
    rrmse = metrics.compute_rrmse(y_true, y_pred)
    cp_time=history['val_inference_time_ms']

    # === Save results ===
    result = {
        'model_id': model_id,
        "pcc": pcc,
        "snr_diff": snr_diff,
        "rmse": rmse,
        "rrmse": rrmse,
        "cp_time": cp_time,}
    
    # Convert the tensors to scalar values using .item() before appending
    all_pcc.append(pcc.item() if isinstance(pcc, torch.Tensor) else pcc)
    all_snr_diff.append(snr_diff.item() if isinstance(snr_diff, torch.Tensor) else snr_diff)
    all_rmse.append(rmse.item() if isinstance(rmse, torch.Tensor) else rmse)
    all_rrmse.append(rrmse.item() if isinstance(rrmse, torch.Tensor) else rrmse)
    all_cpt.append(cp_time)
    
    df_result = pd.DataFrame([result])
    model_results_file_id = os.path.join(model_config_file,f"model{model_id}_results.csv")
    df_result.to_csv(model_results_file_id, mode='a', index=False,
                    header=not os.path.exists(model_results_file_id))

        # # === Plots (optional) ===
    ut.plot_loss(history,model_config_file)
    ut.plot_predictions(y_true, y_pred, x_input, 20, model_config_file)
    fold += 1


# === Summary Statistics ===
pcc_avg = np.mean(all_pcc)
pcc_std = np.std(all_pcc)

snr_diff_avg = np.mean(all_snr_diff)
snr_diff_std = np.std(all_snr_diff)

rmse_avg = np.mean(all_rmse)
rmse_std = np.std(all_rmse)

rrmse_avg = np.mean(all_rrmse)
rrmse_std = np.std(all_rrmse)
                        
cpt_avg = np.mean(all_cpt)
cpt_std = np.std(all_cpt)

all_results = {
"pcc": f"{pcc_avg:.4f} ± {pcc_std:.4f}",
"snr_diff": f"{snr_diff_avg:.4f} ± {snr_diff_std:.4f}",
"rmse": f"{rmse_avg:.4f} ± {rmse_std:.4f}",
"rrmse": f"{rrmse_avg:.4f} ± {rrmse_std:.4f}",
"cpt(ms)": f"{cpt_avg:.4f} ± {cpt_std:.4f}"}

df_result = pd.DataFrame([result])
model_results_file_id = os.path.join(model_config_file,"Final_results.csv")
df_result.to_csv(model_results_file_id, mode='a', index=False,
header=not os.path.exists(model_results_file_id))