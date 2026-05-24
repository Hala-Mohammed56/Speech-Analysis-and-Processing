import librosa
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Load Audio File
# ==========================================
audio_path = "../wav_files/S1-a-rep1.wav"

x, fs = librosa.load(audio_path, sr=None)

print("Sampling Rate:", fs)
print("Audio Duration:", round(len(x)/fs, 3), "seconds")

# ==========================================
# Select Voiced Segment
# ==========================================
start_time = 0.57
end_time = 1.22

start_sample = int(start_time * fs)
end_sample = int(end_time * fs)

segment = x[start_sample:end_sample]

# ==========================================
# Waveform Plot
# ==========================================
time = np.arange(len(segment)) / fs

plt.figure(figsize=(10,4))
plt.plot(time, segment)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# ==========================================
# Autocorrelation
# ==========================================
autocorr = np.correlate(segment, segment, mode='full')
autocorr = autocorr[len(autocorr)//2:]

# Pitch range
min_f0 = 80
max_f0 = 300

min_lag = int(fs / max_f0)
max_lag = int(fs / min_f0)

# Find peak
peak_index = np.argmax(autocorr[min_lag:max_lag]) + min_lag

# F0 estimation
f0_autocorr = fs / peak_index

print("\nEstimated F0 using Autocorrelation:")
print(f"{f0_autocorr:.2f} Hz")

# ==========================================
# Autocorrelation Plot
# ==========================================
lags = np.arange(len(autocorr)) / fs

plt.figure(figsize=(10,4))
plt.plot(lags, autocorr)
plt.title("Autocorrelation")
plt.xlabel("Lag (s)")
plt.ylabel("Correlation")
plt.grid()
plt.show()

# ==========================================
# Pitch Contour using PYIN
# ==========================================
f0, voiced_flag, voiced_probs = librosa.pyin(
    segment,
    fmin=80,
    fmax=300,
    sr=fs
)

times = librosa.times_like(f0, sr=fs)

# Remove NaN values
valid = ~np.isnan(f0)

plt.figure(figsize=(10,4))
plt.plot(times[valid], f0[valid])
plt.title("Pitch Contour")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid()
plt.show()

# ==========================================
# Mean F0 from Pitch Tracking
# ==========================================
mean_f0 = np.nanmean(f0)

print("\nEstimated F0 using Pitch Tracking:")
print(f"{mean_f0:.2f} Hz")

# ==========================================
# Comparison Between Methods
# ==========================================
print("\nComparison Between Methods")
print("--------------------------------")
print(f"Autocorrelation F0 : {f0_autocorr:.2f} Hz")
print(f"Pitch Tracking F0  : {mean_f0:.2f} Hz")