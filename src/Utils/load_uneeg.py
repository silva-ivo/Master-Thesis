import mne

data_base_dir = "/data/datasets/20240201_UNEEG_ForMayo/B52K3P3G/V5a/B52K3P3G_01_0004245_20211008_01_EEGdata.edf"

# preload=True loads the data into memory
raw = mne.io.read_raw_edf(data_base_dir, preload=True, verbose=True)

# Display basic info
print(raw.info)

# Plot (optional, requires a GUI)
# raw.plot()

# Access EEG data as numpy
data, times = raw.get_data(return_times=True)

print("Shape of EEG data:", data.shape)  # (n_channels, n_samples)
