import torch
import torch.nn as nn
import torch.optim as optim
import os



def evalute_model(model,test_loader,device):
    
    model.eval()
    y_pred_tensor = torch.empty(0, *y_pred.shape[1:]).to(device)  
    y_true_tensor = torch.empty(0, *y_test_batch.shape[1:]).to(device)
    x_inputs_tensor = torch.empty(0, *X_test_batch.shape[1:]).to(device)

    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            y_pred = model(X_test_batch)

            y_pred_tensor = torch.cat((y_pred_tensor, y_pred), dim=0)
            y_true_tensor = torch.cat((y_true_tensor, y_test_batch), dim=0)
            x_inputs_tensor = torch.cat((x_inputs_tensor, X_test_batch), dim=0)

    return x_inputs_tensor.cpu(), y_true_tensor.cpu(), y_pred_tensor.cpu()