# 🎧 Rule Based Voice vs Noise Detection

This repository contains a lightweight Python script that segments a one minute audio file into one second intervals and classifies each as either human voice or non speech noise using a fully rule based system with no machine learning.

---

## 📌 Overview

### Goal:
Classify one second audio segments as voice or noise using frequency domain analysis and interpretable rules.

### Signal Processing Pipeline
1. **Audio Segmentation**: Raw audio is divided into 1-second non-overlapping segments
2. **Bandpass Filtering**: 4th-order Butterworth filter (300-1500 Hz) isolates speech frequencies
3. **Frequency Domain Analysis**: Real FFT converts time-domain signals to frequency domain
4. **Feature Extraction**: Compute spectral energy, flatness, pitch, voicing probability, and voice band ratio
5. **Feature Smoothing**: Moving average window (size=3) reduces temporal noise
6. **Rule-Based Classification**: Weighted scoring system with configurable thresholds
---

## 📂 Input

The script takes a mono WAV audio file approximately one minute long that contains a mix of:

• Clear human speech  
• Background noise including crowds, keyboard typing, fans, or ambient hum

You can use a preexisting audio file or create your own using the provided `record.py` script:

```bash
python record.py
```
Audio is segmented into one-second chunks. If the final portion of the file is shorter than 0.2 seconds, it is discarded to avoid processing very short and unreliable segments.

---

## 🎛️ Feature Extraction

Some features are extracted from the raw audio to preserve the natural structure of the signal, while others are computed after applying a fourth-order Butterworth bandpass filter to mimic the human speech range (300 to 1500 Hz). Each segment is then converted into the frequency domain using a Real Fast Fourier Transform, which reveals how energy is distributed across frequencies. This enables the extraction of key features such as spectral energy, flatness, and voice band energy ratio. All features are smoothed using a short moving average window to reduce the impact of sudden spikes or noise fluctuations across adjacent segments. This helps stabilize the final classification and improve overall consistency.


### Total Spectral Energy
Captures the overall power of the signal. Human speech generally carries more energy than ambient noise.

### Spectral Flatness 
Indicates how noise like the signal is. High flatness suggests white noise or static, while low flatness indicates more tonal structure as seen in speech.

### Fundamental Pitch
Reflects the base frequency of vocal fold vibration, a strong marker of human speech.

### Voicing Probability  
Represents the percentage of time the signal is voiced. High voicing means more likelihood of speech being present.

### Voice Band Energy Ratio (Log Normalized)  
Measures how much energy is present in the typical human speech range relative to total energy. The result is log scaled to reduce sensitivity to outliers and better capture proportional differences.

---

## 🧠 Classification Logic

Each segment receives a score based on the extracted features. A segment is labeled as voice only if the combined score passes a configurable threshold.

Scoring rules are based on:  
• Minimum energy  
• Flatness below a threshold  
• Valid pitch range  
• Sufficient voicing probability  
• Voice band energy ratio above threshold

Each feature contributes a weighted number of points toward the final score. Flatness and voice band ratio have greater weight (2 points each), while pitch and voicing probability contribute one point each. This helps prioritize features that most reliably distinguish voice from noise.

---

## 🧪 Output

The script outputs a structured list of labeled segments in results.json:

```json
[
  {"start_time": 0, "end_time": 1, "label": "voice"},
  {"start_time": 1, "end_time": 2, "label": "noise"}
]
```

The terminal also prints a detailed feature summary for each segment, with colored indicators showing which thresholds were met. You can additionally visualize feature trends using the time series plots for deeper analysis.

## 🚀 How to Run

**Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/vad-rule-based.git
cd vad-rule-based
```
**Step 2: Install Python Dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Add Your Audio File**
Place your audio file inside the recordings folder. Example:

```bash
recordings/recording.wav
```
**Step 4: Run the Script**

```bash
python audioSegmentation.py
```
## 📋 Assumptions and Limitations

• Input must be mono audio sampled at 16000 Hz  
• The method is not designed for overlapping speakers or music with vocals  
• Thresholds are empirically tuned and may require adjustment for different environments  
• Built for interpretability and lightweight performance without any machine learning  

## 📬 Contact
Created by Mekhael Thaha
