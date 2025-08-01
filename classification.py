import os
import json
import numpy as np
import librosa
import parselmouth
import matplotlib.pyplot as plt
from scipy.fft import rfft
from scipy.signal import butter, filtfilt
from collections import Counter

# Thresholds 
ENERGY_THRESHOLD = 100        # Minimum spectral energy required (filters out silence/very quiet noise)
PITCH_RANGE = (75, 500)       # Valid fundamental frequency range for human speech (Hz)
FLATNESS_THRESHOLD = 0.4      # Maximum spectral flatness for voice (higher = more noise-like)
VOICING_PROB_THRESHOLD = 0.25 # Minimum voicing probability (percentage of voiced frames)
VB_RATIO_THRESHOLD = -0.35    # Minimum log voice band energy ratio (speech frequency concentration)


# Bandpass filter
def butter_bandpass_filter(data, lowcut=300, highcut=1500, sr=16000, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Segment audio 
def segment_audio(filepath, sr=16000, min_partial_sec=0.2):
    y = librosa.load(filepath, sr=sr, mono=True)[0]
    segment_length = sr
    total_segments = len(y) // segment_length
    segments_raw = [y[i * segment_length : (i + 1) * segment_length] for i in range(total_segments)]
    remainder = y[total_segments * segment_length:]
    if len(remainder) > min_partial_sec * segment_length:
        segments_raw.append(remainder)
    return segments_raw, sr

# Extract pitch and voicing probability 
def extract_parselmouth_features(raw_segment, sr=16000):
    snd = parselmouth.Sound(values=raw_segment, sampling_frequency=sr)
    pitch_obj = snd.to_pitch()
    pitches = pitch_obj.selected_array['frequency']
    avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    voiced_frames = pitches > 0
    voicing_prob = np.sum(voiced_frames) / len(pitches)

    return avg_pitch, voicing_prob

# Extract voice band ratio 
def voice_band_energy_ratio(raw_segment, sr):
    # Use raw signal for total power
    fft_raw = np.abs(rfft(raw_segment))
    freqs = np.fft.rfftfreq(len(raw_segment), d=1/sr)
    total_power = np.sum(fft_raw ** 2) + 1e-10

    # Use bandpass-filtered signal to isolate voice band
    filtered = butter_bandpass_filter(raw_segment)
    fft_filtered = np.abs(rfft(filtered))
    voice_band_power = np.sum(fft_filtered ** 2) + 1e-10

    # Log-ratio for robustness
    log_vbr = np.log10(voice_band_power) - np.log10(total_power)

    return log_vbr



# Feature smoothing 
def smooth_feature(values, window_size=3):
    smoothed = []
    for i in range(len(values)):
        window = values[max(0, i - window_size//2):min(len(values), i + window_size//2 + 1)]
        smoothed.append(np.mean(window))
    return smoothed

# Extracts all features at once
def extract_features(raw_segment, sr=16000):
    
    # Apply bandpass filter and convert to frequency domain using FFT
    filtered = butter_bandpass_filter(raw_segment, sr=sr)
    fft_mag = np.abs(rfft(filtered))
    total_energy = np.sum(fft_mag ** 2)  

    # Calculate spectral flatness
    raw_fft_mag = np.abs(rfft(raw_segment))
    geometric_mean = np.exp(np.mean(np.log(raw_fft_mag + 1e-10)))
    arithmetic_mean = np.mean(raw_fft_mag + 1e-10)
    spectral_flatness = geometric_mean / arithmetic_mean  

    # Extract pitch and voicing information
    pitch, voicing_prob = extract_parselmouth_features(raw_segment, sr)
    
    # Calculate voice band energy ratio
    vb_ratio = voice_band_energy_ratio(raw_segment, sr)


    return {
        "total_energy": total_energy,
        "spectral_flatness": spectral_flatness,
        "pitch": pitch,
        "voicing_prob": voicing_prob,
        "voice_band_ratio": vb_ratio
    }

# Classification scoring system
def classify_segment(features):
    if features["total_energy"] < ENERGY_THRESHOLD:
        return "noise"

    score = 0
    if features["spectral_flatness"] < FLATNESS_THRESHOLD:
        score += 2
    if PITCH_RANGE[0] <= features["pitch"] <= PITCH_RANGE[1]:
        score += 1
    if features["voicing_prob"] > VOICING_PROB_THRESHOLD:
        score += 1
    if features["voice_band_ratio"] > VB_RATIO_THRESHOLD:
        score += 2

    return "voice" if score >= 4 else "noise"

# Plotting features
def plot_features(times, energies, flatnesses, pitches, voicing_probs, vb_ratios):
    plt.figure(figsize=(14, 14))

    plt.subplot(5, 1, 1)
    plt.plot(times, energies, marker='o', label="Smoothed Energy")
    plt.axhline(ENERGY_THRESHOLD, color='r', linestyle='--')
    plt.ylabel("Energy")
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(times, flatnesses, marker='o', label="Smoothed Flatness", color='c')
    plt.axhline(FLATNESS_THRESHOLD, color='r', linestyle='--')
    plt.ylabel("Flatness")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(times, pitches, marker='o', label="Smoothed Pitch", color='g')
    plt.axhline(PITCH_RANGE[0], color='gray', linestyle='--')
    plt.axhline(PITCH_RANGE[1], color='gray', linestyle='--')
    plt.ylabel("Pitch (Hz)")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(times, voicing_probs, marker='o', label="Smoothed Voicing Prob", color='orange')
    plt.axhline(VOICING_PROB_THRESHOLD, color='gray', linestyle='--')
    plt.ylabel("Voicing Prob")
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(times, vb_ratios, marker='o', label="Smoothed Voice Band Ratio", color='brown')
    plt.axhline(VB_RATIO_THRESHOLD, color='gray', linestyle='--')
    plt.ylabel("VBR")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Export results 
def export_results(labels):
    results = [{"start_time": i, "end_time": i + 1, "label": labels[i]} for i in range(len(labels))]
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

# Debug print in terminal
def print_segment_debug_info(labels, energies, flatnesses, pitches, voicing_probs, vb_ratios):
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    print("\n=== Segment Classification Summary ===\n")
    for i in range(len(labels)):
        pitch_pass = GREEN if PITCH_RANGE[0] <= pitches[i] <= PITCH_RANGE[1] else RED
        flatness_pass = GREEN if flatnesses[i] < FLATNESS_THRESHOLD else RED
        voicing_pass = GREEN if voicing_probs[i] > VOICING_PROB_THRESHOLD else RED
        vbr_pass = GREEN if vb_ratios[i] > VB_RATIO_THRESHOLD else RED
        energy_pass = GREEN if energies[i] >= ENERGY_THRESHOLD else RED

        print(f"Second {i}:")
        print(f"  Label:           {labels[i].upper()}")
        print(f"  Energy:          {energy_pass}{energies[i]:.2f}{RESET}")
        print(f"  Flatness:        {flatness_pass}{flatnesses[i]:.3f}{RESET}")
        print(f"  Pitch:           {pitch_pass}{pitches[i]:.1f} Hz{RESET}")
        print(f"  Voicing Prob:    {voicing_pass}{voicing_probs[i]:.3f}{RESET}")
        print(f"  Voice Band Ratio:{vbr_pass}{vb_ratios[i]:.3f}{RESET}")
        print("-" * 40)


if __name__ == "__main__":
    # Load and segment audio
    filepath = "recordings/recording.wav"
    segments_raw, sr = segment_audio(filepath)

    # Filter, convert to frequency domain and extract features 
    energies, flatnesses, pitches, voicing_probs, vb_ratios = [], [], [], [], []
    for segment in segments_raw:
        features = extract_features(segment, sr)
        energies.append(features["total_energy"])
        flatnesses.append(features["spectral_flatness"])
        pitches.append(features["pitch"])
        voicing_probs.append(features["voicing_prob"])
        vb_ratios.append(features["voice_band_ratio"])

    # Temporally smoothe features
    sm_energies = smooth_feature(energies)
    sm_flatnesses = smooth_feature(flatnesses)
    sm_pitches = smooth_feature(pitches)
    sm_voicing_probs = smooth_feature(voicing_probs)
    sm_vb_ratios = smooth_feature(vb_ratios)

    # Classify features
    smoothed_labels = []
    for i in range(len(sm_energies)):
        smoothed_features = {
            "total_energy": sm_energies[i],
            "spectral_flatness": sm_flatnesses[i],
            "pitch": sm_pitches[i],
            "voicing_prob": sm_voicing_probs[i],
            "voice_band_ratio": sm_vb_ratios[i]
        }
        smoothed_labels.append(classify_segment(smoothed_features))

    # Terminal debug logs
    print_segment_debug_info(smoothed_labels, sm_energies, sm_flatnesses, sm_pitches, sm_voicing_probs, sm_vb_ratios)
    export_results(smoothed_labels)

    # Plots
    times = list(range(len(smoothed_labels)))
    plot_features(times, sm_energies, sm_flatnesses, sm_pitches, sm_voicing_probs, sm_vb_ratios)
