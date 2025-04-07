import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from sklearn.model_selection import KFold, GroupKFold
from Utils import utils

# Define EEGDataset and PatientBatchSampler classes for NESTED CV
class EEGDataset(Dataset):
    def __init__(self, inputs, targets, patient_ids):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.patient_ids = np.array(patient_ids)  # Track patient IDs for batching

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.patient_ids[idx]

class PatientBatchSampler(Sampler):
    def __init__(self, patient_ids, batch_size):
        self.batch_size = batch_size
        self.patient_windows = {}
        
        # Group indices by patient
        for idx, pid in enumerate(patient_ids):
            if pid not in self.patient_windows:
                self.patient_windows[pid] = []
            self.patient_windows[pid].append(idx)
        
        self.patient_list = list(self.patient_windows.keys())
        self.batches = self._create_batches()
    
    def _create_batches(self):
        np.random.shuffle(self.patient_list)  # Shuffle patient order each epoch
        
        patient_iters = {pid: iter(self.patient_windows[pid]) for pid in self.patient_list}  # Create iterators
        batches = []
        temp_batch = []

        while patient_iters:
            for patient in list(patient_iters.keys()):  # Iterate over patients in shuffled order
                try:
                    temp_batch.append(next(patient_iters[patient]))  # Add one window per patient
                    if len(temp_batch) == self.batch_size:
                        batches.append(temp_batch)
                        temp_batch = []
                except StopIteration:
                    del patient_iters[patient]  # Remove patients when their windows are exhausted
        
        if temp_batch:  # Handle any remaining windows
            batches.append(temp_batch)

        return batches
    
    def __iter__(self):
        np.random.shuffle(self.batches)  # Shuffle batches each epoch
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)

def load_nested_cv_patients(data_base_dir, window_size, batch_size, outer_folds, inner_folds):


    patient_folders = sorted(glob.glob(os.path.join(data_base_dir, "Filtered_Dados_paciente*")))
    if not patient_folders:
        raise FileNotFoundError("No patient folders found in Data directory.")
    
    all_inputs, all_targets, patient_ids = [], [], []
    
    for patient_folder in patient_folders:
        patient_id = os.path.basename(patient_folder)
        input_files = sorted(glob.glob(os.path.join(patient_folder, "original_filtered_segment_*.npy")))
        target_files = sorted(glob.glob(os.path.join(data_base_dir, patient_id.replace("Filtered_", ""), "preprocessed_segment_*.npy")))
        
        if not input_files or not target_files:
            continue
        
             
        inputs= [np.load(f) for f in input_files]
        targets = [np.load(f) for f in target_files]

        X_patient, y_patient = utils.split_segments(np.array(inputs),np.array(targets) , window_size)
        
        all_inputs.append(X_patient)
        all_targets.append(y_patient)
        patient_ids.extend([patient_id] * len(X_patient))  # Assign patient ID to each window
    
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    outer_cv = GroupKFold(n_splits=outer_folds)
    
    for outer_train_idx, outer_test_idx in outer_cv.split(patient_folders, groups=patient_folders):
        test_patients = [patient_folders[i] for i in outer_test_idx]
        train_patients = [patient_folders[i] for i in outer_train_idx]
        
        train_mask = np.isin(patient_ids, [os.path.basename(p) for p in train_patients])
        test_mask = np.isin(patient_ids, [os.path.basename(p) for p in test_patients])
        
        X_train, y_train, p_train = all_inputs[train_mask], all_targets[train_mask], np.array(patient_ids)[train_mask]
        X_test, y_test = all_inputs[test_mask], all_targets[test_mask]
        
        inner_cv = GroupKFold(n_splits=inner_folds)
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train, groups=p_train):

            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
            p_inner_train, p_inner_val = p_train[inner_train_idx], p_train[inner_val_idx]
            
            train_dataset = EEGDataset(X_inner_train, y_inner_train, p_inner_train)
            val_dataset = EEGDataset(X_inner_val, y_inner_val, p_inner_val)
            test_dataset = EEGDataset(X_test, y_test, np.array(["test"] * len(y_test)))
            
            train_loader = DataLoader(train_dataset, batch_sampler=PatientBatchSampler(p_inner_train, batch_size), pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            
            yield train_loader, val_loader, test_loader

#Simpler dataloader

def get_dataloaders(data_base_dir, window_size, batch_size=32, split_ratio=(0.7, 0.15, 0.15)):
    
    patient_folders = sorted(glob.glob(os.path.join(data_base_dir, "Filtered_Dados_paciente*")))
    if not patient_folders:
        raise FileNotFoundError("No patient folders found in Data directory.")
    
    all_inputs, all_targets = [], []
    
    
    
    
    for patient_folder in patient_folders:
        patient_id = os.path.basename(patient_folder)
        input_files = sorted(glob.glob(os.path.join(patient_folder, "original_filtered_segment_*.npy")))
        target_files = sorted(glob.glob(os.path.join(data_base_dir, patient_id.replace("Filtered_", ""), "preprocessed_segment_*.npy")))
        
        if not input_files or not target_files:
            continue
        
        
        inputs= [np.load(f) for f in input_files]
        targets = [np.load(f) for f in target_files]

        X_patient, y_patient = utils.split_segments(np.array(inputs),np.array(targets) , window_size)
        all_inputs.append(X_patient)
        all_targets.append(y_patient)

        
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    
    dataset = utils.TimeSeriesDataset(torch.tensor(all_inputs, dtype=torch.float32), 
                                          torch.tensor(all_targets, dtype=torch.float32))

    train_size = int(split_ratio[0] * len(dataset))
    val_size = int(split_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, train_size + val_size + test_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader