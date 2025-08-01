# 🎧 Rule Based Voice vs Noise Detection

This repository contains a lightweight Python script that segments a one minute audio file into one second intervals and classifies each as either human voice or non speech noise using a fully rule based system with no machine learning required.

This project simulates a lightweight audio pre processor suitable for filtering scientific voice data or embedded voice logs.

---

## 📌 Overview

**Goal:**  
Classify one second audio segments as voice or noise using frequency domain analysis and interpretable rules.

**Key Features**  
• Log normalized voice band energy ratio  
• Spectral energy  
• Spectral flatness  
• Fundamental pitch estimation using Parselmouth  
• Voicing probability

---

## 📂 Input

The script takes a mono WAV audio file approximately one minute long that contains a mix of:

• Clear human speech  
• Background noise including crowds, keyboard typing, fans, or ambient hum

You can use a preexisting audio file or create your own using the provided `record.py` script:

```bash
python record.py
```

---

## 🎛️ Feature Extraction

Each one second segment is analyzed using the following features to identify the presence of voice:

**Total Spectral Energy**  
Captures the overall power of the signal. Human speech generally carries more energy than ambient noise.

**Spectral Flatness**  
Indicates how noise like the signal is. High flatness suggests white noise or static, while low flatness indicates more tonal structure as seen in speech.

**Fundamental Pitch (F₀)**  
Reflects the base frequency of vocal fold vibration, a strong marker of human speech.

**Voicing Probability**  
Represents the percentage of time the signal is voiced. High voicing means more likelihood of speech being present.

**Voice Band Energy Ratio (Log Normalized)**  
Measures how much energy is present in the typical human speech range between 300 and 3400 Hz, relative to total energy. The result is log scaled to reduce sensitivity to outliers and better capture proportional differences.

---

## 🧠 Classification Logic

Each segment receives a score based on the extracted features. A segment is labeled as voice only if the combined score passes a configurable threshold.

Scoring rules are based on:  
• Minimum energy  
• Flatness below a threshold  
• Valid pitch range  
• Sufficient voicing probability  
• Voice band energy ratio above threshold

These rules are tuned for real world recordings.

---

## 🧪 Output

The script outputs a structured list of labeled segments in JSON format:

```json
[
  {"start_time": 0, "end_time": 1, "label": "voice"},
  {"start_time": 1, "end_time": 2, "label": "noise"}
]
```

The terminal also prints a detailed feature summary for each segment with colored indicators showing which thresholds were met and you can visualize feature trends using the time series plots for deeper analysis.

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
