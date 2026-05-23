import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.fft import fft
from scipy.signal import freqz
import os


def lpc_coefficients(x, order):
    
    x = x - np.mean(x)
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    R = autocorr[:order + 1]

    # Levinson-Durbin
    a = np.zeros(order + 1)
    e = R[0]
    a[0] = 1.0

    for i in range(1, order + 1):
        acc = R[i]
        for j in range(1, i):
            acc += a[j] * R[i - j]

        k = -acc / e
        a_new = a.copy()
        a_new[i] = k

        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]

        a = a_new
        e *= (1 - k * k)

    return a


def analyze_audio(filename, sound_label):
    # Read audio
    x, fs = librosa.load(filename, sr=None, mono=True)

    # Normalize
    x = x / np.max(np.abs(x))

    # Remove silence
    x_trimmed, _ = librosa.effects.trim(x, top_db=25)
    x = x_trimmed

    # Take middle stable part
    duration = len(x) / fs
    start = int(0.25 * len(x))
    end = int(0.75 * len(x))
    x_segment = x[start:end]

    output_folder = "part3_figures"
    os.makedirs(output_folder, exist_ok=True)


    # 1. Waveform
    t = np.arange(len(x_segment)) / fs

    plt.figure(figsize=(8, 4))
    plt.plot(t, x_segment)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform of {sound_label}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{sound_label}_waveform.png", dpi=300)
    plt.show()


    # 2. FFT Spectrum
    N = len(x_segment)
    window = np.hamming(N)
    x_win = x_segment * window

    X = np.abs(fft(x_win))
    freqs = np.fft.fftfreq(N, d=1/fs)

    half = N // 2
    freqs = freqs[:half]
    X = X[:half]

    X_db = 20 * np.log10(X + 1e-8)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, X_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"FFT Spectrum of {sound_label}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{sound_label}_fft_spectrum.png", dpi=300)
    plt.show()


    # 3. Spectral Envelope
    envelope = librosa.feature.melspectrogram(
        y=x_segment,
        sr=fs,
        n_fft=1024,
        hop_length=256,
        n_mels=40
    )

    envelope_db = librosa.power_to_db(envelope, ref=np.max)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        envelope_db,
        sr=fs,
        hop_length=256,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectral Envelope of {sound_label}")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{sound_label}_spectral_envelope.png", dpi=300)
    plt.show()

    # 4. LPC Analysis
    order = 12
    a = lpc_coefficients(x_segment, order)

    w, h = freqz(1, a, worN=1024, fs=fs)
    h_db = 20 * np.log10(np.abs(h) + 1e-8)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, X_db, label="FFT Spectrum")
    plt.plot(w, h_db, label="LPC Envelope", linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"LPC Analysis of {sound_label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{sound_label}_lpc_analysis.png", dpi=300)
    plt.show()


# Analyze vowel /a/
analyze_audio("female_a.wav", "vowel_a")

# Analyze fricative /s/
analyze_audio("female_s.wav", "fricative_s")