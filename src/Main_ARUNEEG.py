# %%
# %%
import os
import glob
import math
from datetime import datetime
import numpy as np
import torch
import pyedflib
import plotly.graph_objects as go

import models.DAE_models
import models.cnn_models  # make sure your model module is in the path

def main(sz2plot):
    data_dir = "/data/home/silva/TrainingData/TrainingP1"
    model_path = "/data/home/silva/Documents/Pipline_2/Results/Final_Validation/pat_112802_model/best_model.pth"
    #model_path="/data/home/silva/Documents/Pipline_2/Results/SE_ResNet1D/Phase_5_Final_Validation/modelmodel0best_model.pth"
    # Load model
    #model= models.DAE_models.SE_UNet_5()
    model = models.cnn_models.SE_ResNet1D(2, 5, [32, 64, 128, 256, 512], [9, 7, 5, 3, 3], 4, True, 0.1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    files = glob.glob(os.path.join(data_dir, '*.edf'))
    files.sort()

    model_len_sec = 5
    plot_len_sec = 10
    nsz_found = 0

    for fn in files:
        try:
            f = pyedflib.EdfReader(fn)
          
            for on in sz2plot:
                on_ts = datetime.strptime(on, '%Y-%m-%d %H:%M:%S')
                on_unix = on_ts.timestamp()
                start_ts = f.getStartdatetime()
                start_unix = start_ts.timestamp()
                duration_sec = math.floor(f.file_duration)

                if start_unix <= on_unix < (start_unix + duration_sec):
                    print(f"\n-------------------- Seizure found at: {on} --------------------")
                    print("File: " + fn)
                    
                    n = f.signals_in_file
                    signal_labels = f.getSignalLabels()
                    nsamples = f.getNSamples()[0]
                    duration_sec = math.floor(f.file_duration)
                    sampling_freq = nsamples / duration_sec

                    print(f"[INFO] Processing file: {fn}")
                    print(f"[INFO] Sampling rate: {sampling_freq:.2f} Hz")

                    sigbufs = np.zeros((n, nsamples))
                    for i in range(n):
                        sigbufs[i, :] = f.readSignal(i)

                    expected_model_len = int(model_len_sec * sampling_freq)
                    expected_plot_len = int(plot_len_sec * sampling_freq)
                            
                    ix_sz = int((on_ts - start_ts).total_seconds() * sampling_freq)

                    window_offsets_sec = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,65,65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]

                    for i, offset in enumerate(window_offsets_sec):
                        plot_start_idx = int(ix_sz + offset * sampling_freq)
                        plot_end_idx = plot_start_idx + expected_plot_len

                        plot_start_idx = max(0, plot_start_idx)
                        plot_end_idx = min(sigbufs.shape[1], plot_end_idx)
                        seg_len = plot_end_idx - plot_start_idx

                        x_vals = np.arange(0, seg_len) / sampling_freq
                        segment_plot = sigbufs[:, plot_start_idx:plot_end_idx]

                        cleaned_all = []

                        for seg_i in range(0, seg_len, expected_model_len):
                            seg_start = plot_start_idx + seg_i
                            seg_end = seg_start + expected_model_len

                            if seg_end > sigbufs.shape[1]:
                                break  # Avoid incomplete last segment

                            segment_model = sigbufs[:, seg_start:seg_end]
                            segment_t = segment_model.T
                            input_tensor = torch.tensor(segment_t, dtype=torch.float32).unsqueeze(0)

                            with torch.no_grad():
                                cleaned_seg = model(input_tensor).squeeze(0).T.numpy()

                            cleaned_all.append(cleaned_seg)

                        if not cleaned_all:
                            continue

                        cleaned_full = np.concatenate(cleaned_all, axis=1)
                        
                        

                        for ch in range(2):  # Plot channels 0 and 1 separately
                            fig = go.Figure()

                            # Set color depending on channel
                            raw_color = "grey" if ch == 0 else "grey"
                            cleaned_color = "blue" if ch == 0 else "green"

                            fig.add_trace(go.Scatter(
                                x=x_vals,
                                y=segment_plot[ch, :seg_len],
                                mode='lines',
                                name=f"{signal_labels[ch]} Raw",
                                line=dict(color=raw_color, width=0.5)
                            ))
                            fig.add_trace(go.Scatter(
                                x=x_vals[:cleaned_full.shape[1]],
                                y=cleaned_full[ch, :cleaned_full.shape[1]],
                                mode='lines',
                                name=f"{signal_labels[ch]} Cleaned",
                                line=dict(color=cleaned_color, width=1.5)
                            ))

                            if plot_start_idx <= ix_sz <= plot_end_idx:
                                seizure_time_sec = (ix_sz - plot_start_idx) / sampling_freq
                                fig.add_trace(go.Scatter(
                                    x=[seizure_time_sec, seizure_time_sec],
                                    y=[-100, 100],
                                    mode='lines',
                                    line=dict(color="red", width=2),
                                    name="Seizure Onset"
                                ))

                            fig.update_layout(
                                title=f"Channel {ch} | Offset {offset}s | Seizure near",
                                width=900,
                                height=400,
                                yaxis=dict(range=[-100, 100]),
                                legend=dict(orientation="h")
                            )
                            fig.show()
                    nsz_found += 1
                    if nsz_found == len(sz2plot):
                        print("\n[INFO] All seizure timestamps processed.")
                        break

            f.close()

        except Exception as e:
            print(f">>>>> ERROR processing file {fn}: {e} <<<<<")
            try:
                f.close()
            except:
                pass

if __name__ == "__main__":
    sz2plot = ['2020-12-05 06:27:23']  # Replace with your timestamps
    main(sz2plot)
