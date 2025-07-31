import os
import json
import numpy as np
import librosa
import parselmouth
import matplotlib.pyplot as plt
from scipy.fft import rfft
from scipy.signal import butter, filtfilt
from collections import Counter

# === Thresholds for rule-based classification ===
ENERGY_THRESHOLD = 100
PITCH_RANGE = (60, 400)      # Human pitch range in Hz
HNR_THRESHOLD = 5            # dB threshold for harmonicity

# === Bandpass Filter (used for energy computation) ===
def butter_bandpass_filter(data, lowcut=300, highcut=1500, sr=16000, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# === Audio Segmentation ===
def segment_audio(filepath, sr=16000, min_partial_sec=0.2):
    """Split audio into 1-second non-overlapping segments."""
    y = librosa.load(filepath, sr=sr, mono=True)[0]
    segment_length = sr
    total_segments = len(y) // segment_length

    segments_raw = [y[i * segment_length : (i + 1) * segment_length] for i in range(total_segments)]
    remainder = y[total_segments * segment_length:]
    if len(remainder) > min_partial_sec * segment_length:
        segments_raw.append(remainder)
    return segments_raw, sr

# === Extract pitch and HNR using Parselmouth ===
def extract_pitch_and_hnr_parselmouth(raw_segment, sr=16000):
    snd = parselmouth.Sound(values=raw_segment, sampling_frequency=sr)

    pitch_obj = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    harmonicity_obj = snd.to_harmonicity_cc()

    pitches = pitch_obj.selected_array['frequency']
    voiced_pitches = pitches[pitches > 0]
    avg_pitch = np.mean(voiced_pitches) if len(voiced_pitches) > 0 else 0

    hnr_values = harmonicity_obj.values[0]
    hnr_values = np.array(hnr_values)
    hnr_values = hnr_values[~np.isnan(hnr_values)]
    hnr_values = hnr_values[hnr_values > 0]
    avg_hnr = np.mean(hnr_values) if len(hnr_values) > 0 else 0

    return avg_pitch, avg_hnr

# === Feature Extraction ===
def extract_features(raw_segment, sr=16000):
    """Compute energy, pitch, and HNR from segment."""
    filtered = butter_bandpass_filter(raw_segment, sr=sr)
    fft_mag = np.abs(rfft(filtered))
    total_energy = np.sum(fft_mag ** 2)

    pitch, hnr = extract_pitch_and_hnr_parselmouth(raw_segment, sr)

    return {
        "total_energy": total_energy,
        "pitch": pitch,
        "hnr": hnr
    }

# === Rule-Based Classification ===
def classify_segment(features):
    if features["total_energy"] < ENERGY_THRESHOLD:
        return "noise"
    if features["hnr"] > HNR_THRESHOLD and PITCH_RANGE[0] <= features["pitch"] <= PITCH_RANGE[1]:
        return "voice"
    return "noise"

# === Majority-Vote Smoothing ===
def smooth_labels(labels, window_size=3):
    smoothed = []
    for i in range(len(labels)):
        window = labels[max(0, i - window_size//2):min(len(labels), i + window_size//2 + 1)]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed

# === Plot Features Over Time ===
def plot_features(times, total_energies, pitches, hnrs):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(times, total_energies, marker='o', label='Total Energy')
    plt.axhline(ENERGY_THRESHOLD, color='r', linestyle='--', label='Energy Threshold')
    plt.ylabel("Total Energy")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, pitches, marker='o', label='Pitch (Hz)', color='g')
    plt.axhline(PITCH_RANGE[0], color='gray', linestyle='--')
    plt.axhline(PITCH_RANGE[1], color='gray', linestyle='--')
    plt.ylabel("Pitch (Hz)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, hnrs, marker='o', label='HNR (dB)', color='m')
    plt.axhline(HNR_THRESHOLD, color='gray', linestyle='--')
    plt.ylabel("HNR (dB)")
    plt.xlabel("Segment (sec)")
    plt.legend()

    plt.tight_layout()
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    filepath = "recordings/recording.wav"
    segments_raw, sr = segment_audio(filepath)

    total_energies = []
    pitches = []
    hnrs = []
    raw_labels = []

    for segment in segments_raw:
        features = extract_features(segment, sr)
        label = classify_segment(features)

        total_energies.append(features["total_energy"])
        pitches.append(features["pitch"])
        hnrs.append(features["hnr"])
        raw_labels.append(label)

    smoothed_labels = smooth_labels(raw_labels, window_size=3)

    print("Raw Labels:", raw_labels)
    print("Smoothed Labels:", smoothed_labels)

    # === Export JSON Results ===
    results = [
        {"start_time": i, "end_time": i + 1, "label": smoothed_labels[i]}
        for i in range(len(smoothed_labels))
    ]
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # === Optional Visualization ===
    times = list(range(len(segments_raw)))
    plot_features(times, total_energies, pitches, hnrs)
