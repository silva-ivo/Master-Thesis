import torch
import models.cnn_models
import trainer
import Utils.dataloader as dl
import models
import test.tester
import Utils.metrics as metrics



# Parameters
num_epochs = 50
early_stopping_patience = 5
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001


all_pcc, all_snr_diff, all_rmse, all_rrmse = [], [], [], []


# Loop over nested CV splits
for outer_fold, (train_loader, val_loader, test_loader) in enumerate(
    dl.load_nested_cv_patients(
        data_base_dir="/path/to/data",  # Specify your data directory
        window_size=128,  # Example window size
        batch_size=32,  # Example batch size
        outer_folds=5,  # Example number of outer folds
        inner_folds=3   # Example number of inner folds
    )):
    print(f"Starting Outer Fold {outer_fold + 1}")

        # Initialize model
    model = models.cnn_models()  # Replace with your actual model
    model.to(device)

        # Train model using the provided loaders
    trained_model, history, X_val_tensor, y_true_val_tensor, y_pred_val_tensor = trainer.train_model(
        model=model,
        model_name=f"Model_outer{outer_fold}",
        train_loader=train_loader,
        val_loader=val_loader,
        window_size=100,  # Adjust as needed
        loss_function=torch.nn.MSELoss(),  # Example loss function
        device=device,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        lr=learning_rate
        )

        # Evaluate model
    X_test_tensor, y_true_test_tensor, y_pred_test_tensor = test.tester.evalute_model(model,test_loader,device)

    pcc = metrics.compute_pcc(y_true_test_tensor, y_pred_test_tensor)
    snr_diff = metrics.compute_snr_diff(y_true_test_tensor, y_pred_test_tensor, X_test_tensor)
    rmse = metrics.compute_rmse(y_true_test_tensor, y_pred_test_tensor)
    rrmse = metrics.compute_rrmse(y_true_test_tensor, y_pred_test_tensor)

    all_pcc.append(pcc)
    all_snr_diff.append(snr_diff)
    all_rmse.append(rmse)
    all_rrmse.append(rrmse)


    print(f"Outer Fold {outer_fold + 1} completed")
        