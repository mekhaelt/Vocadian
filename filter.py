import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

# 🎚️ Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut=300, highcut=1500, sr=16000, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# 📊 Plot full-audio FFT before/after filter
def plot_full_spectrum_comparison(original, filtered, sr):
    freqs = rfftfreq(len(original), 1 / sr)
    orig_fft = np.abs(rfft(original))
    filt_fft = np.abs(rfft(filtered))

    plt.figure(figsize=(12, 5))
    plt.suptitle("Full Audio - Frequency Spectrum")

    plt.subplot(1, 2, 1)
    plt.plot(freqs, orig_fft)
    plt.title("Before Filter")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 4000)

    plt.subplot(1, 2, 2)
    plt.plot(freqs, filt_fft)
    plt.title("After Butterworth Filter (300–1500 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 4000)

    plt.tight_layout()
    plt.show()

# 📊 Plot full-audio spectrogram before/after filter
def plot_full_spectrogram_comparison(original, filtered, sr):
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
    D_filt = librosa.amplitude_to_db(np.abs(librosa.stft(filtered)), ref=np.max)

    plt.figure(figsize=(12, 5))
    plt.suptitle("Full Audio - Spectrogram")

    plt.subplot(1, 2, 1)
    librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz')
    plt.title("Before Filter")
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(1, 2, 2)
    librosa.display.specshow(D_filt, sr=sr, x_axis='time', y_axis='hz')
    plt.title("After Butterworth Filter (300–1500 Hz)")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt.show()

# 🚀 Main
if __name__ == "__main__":
    filepath = "recordings/recording.wav"

    # Load audio
    y, sr = librosa.load(filepath, sr=16000, mono=True)

    # Apply Butterworth filter
    y_filtered = butter_bandpass_filter(y, sr=sr)

    # Plot full FFT comparison
    plot_full_spectrum_comparison(y, y_filtered, sr)

    # Plot full spectrogram comparison
    plot_full_spectrogram_comparison(y, y_filtered, sr)
