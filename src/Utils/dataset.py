import os
import numpy as np
import utils
import glob
import preprocessing




data_base_dir = "/data/home/silva/Documents/Pipline_2/Data"


patient_folders = sorted(glob.glob(os.path.join(data_base_dir, "Dados_paciente*")))

if not patient_folders:
    raise FileNotFoundError(" No patient folders found in Data directory.")

for patient_folder in patient_folders:
    patient_id = os.path.basename(patient_folder)  # Extract patient name (e.g., "Dados_paciente402")
    patient_output_dir = os.path.join(data_base_dir, f"Filtered_{patient_id}")

    os.makedirs(patient_output_dir, exist_ok=True)
    print(f" Processing {patient_id}...")

    original_segment = utils.load_data("original_segment_*.npy", patient_folder)
    target_segment = utils.load_data("preprocessed_segment_*.npy", patient_folder)

    print(f" {patient_id}: Original Data Shape: {original_segment.shape}")  # Expected: (162, 15360, 19)
    print(f" {patient_id}: Target Data Shape: {target_segment.shape}")

    check_path = os.path.join(patient_output_dir, "original_filtered_segment_001.npy")

    if os.path.exists(check_path):
        print(f"Preprocessed data found for {patient_id}. Skipping preprocessing...")

    else:
        segments_reshaped = original_segment.reshape(-1, 19)
        filtered_segments = preprocessing.preprocess_eeg(segments_reshaped, fs=1000)
        original_filtered_segment = filtered_segments.reshape(original_segment.shape)

        for i in range(original_filtered_segment.shape[0]):  # Loop through all 162 segments
            segment_path = os.path.join(patient_output_dir, f"original_filtered_segment_{i+1:03d}.npy")
            np.save(segment_path, original_filtered_segment[i])

            print(f" {i} segment saved in {patient_output_dir}/")
    
     

















        

