from Utils import dataloader
from Utils import utils as ut
from Utils import metrics   
from models import cnn_models
from train import trainer
import torch

#LOADERS
data_base_dir = "/data/home/silva/Documents/Pipline_2/Data"
window_size = 15360
window_size_name = "1m"
batch_size = 32
#window_size = 15360 1minuto
#windows_size = 7690 30segundos
#windows_size = 2560 10segundos
#window_szie = 1280 5segundos

train_loader, val_loader, test_loader = dataloader.get_dataloaders(data_base_dir, window_size, batch_size)


#TRAINING
loss_function = ut.RRMSELoss()
loss_function_name = "RRMSELoss"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
num_epochs = 40
early_stopping_patience = 5
lr = 0.001

model,history,x_input,y_true,y_pred = trainer.train_model(cnn_models.RestNet_1D_DCNN(19), "RestNet_1D_DCNN", train_loader, val_loader, window_size_name,
                                                          loss_function,loss_function_name,device,num_epochs, early_stopping_patience,lr)


#METRICS
pcc = metrics.compute_pcc(y_true, y_pred)
snr_diff = metrics.compute_snr_diff(y_true, y_pred, x_input)
rmse = metrics.compute_rmse(y_true, y_pred)
rrmse = metrics.compute_rrmse(y_true, y_pred)

metrcis_summary= metrics.compute_metrics_summary(pcc, snr_diff, rmse, rrmse)
metrics.save_metrics_to_csv(metrcis_summary, "RestNet_1D_DCNN",loss_function_name, window_size_name, "training")

#PLOTS
ut.plot_loss(history,loss_function_name, "training", "RestNet_1D_DCNN",loss_function_name, window_size_name)
ut.plot_predictions(y_true, y_pred, x_input, 10, loss_function_name,"training", "RestNet_1D_DCNN", window_size_name)

