import torch

import torch.optim as optim
import os



def train_model(model,model_name, train_loader, val_loader,window_size_name, loss_function,loss_function_name, device, num_epochs, early_stopping_patience,lr):
    print(f"Estou no treino")
    model_save_path=f"/data/home/silva/Documents/Pipline_2/Data/Results/DCNn_GridSearch/Phase_1/{model_name}"
    os.makedirs(model_save_path, exist_ok=True)  
    save_path_bestmodel = os.path.join(model_save_path, "best_model.pth")
        
    
    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr)

    best_val_loss = float('inf')
    patience_counter = early_stopping_patience

    history = {'loss': [], 'val_loss': []}
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch,idx_batch in train_loader:
            X_batch, y_batch = X_batch.to(device,non_blocking=True), y_batch.to(device,non_blocking=True)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()
            

        
        epoch_loss =train_loss/ len(train_loader)
      
        history['loss'].append(epoch_loss)

      
        #Validation
        

        y_pred_tensor = torch.empty(0, *y_pred.shape[1:]).to(device)  
        y_true_tensor = torch.empty(0, *y_pred.shape[1:]).to(device)
        x_inputs_tensor = torch.empty(0, *y_pred.shape[1:]).to(device)
        
        model.eval()
        val_loss = 0.0
        len_val_loss=0
        with torch.no_grad():
            for X_val, y_val, idx_batch in val_loader:
                X_val, y_val = X_val.to(device,non_blocking=True), y_val.to(device,non_blocking=True)

                y_pred = model(X_val)
                loss = criterion(y_pred, y_val)
                
                if loss.item() <1e3:
                    val_loss += loss.item()
                    len_val_loss+=1

                y_pred_tensor = torch.cat((y_pred_tensor, y_pred), dim=0)
                y_true_tensor = torch.cat((y_true_tensor, y_val), dim=0)
                x_inputs_tensor = torch.cat((x_inputs_tensor, X_val), dim=0)
        
            epoch_val_loss = val_loss / len_val_loss
            
            history['val_loss'].append(epoch_val_loss)

        #Early Stopping
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path_bestmodel)
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered. Restoring best model weights.")
            break

    model.load_state_dict(torch.load(save_path_bestmodel))


    return model, history, x_inputs_tensor.cpu(), y_true_tensor.cpu(), y_pred_tensor.cpu()


        