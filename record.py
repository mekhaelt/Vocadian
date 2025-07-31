import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio(duration_sec=10, samplerate=16000):
    print(f"🎙️ Recording for {duration_sec} seconds at {samplerate} Hz...")

    # Record audio
    audio = sd.rec(int(duration_sec * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()

    # Save to hardcoded path
    recordings_folder = r"C:/Users/mekha/Desktop/Projects/Vocadian/recordings"
    os.makedirs(recordings_folder, exist_ok=True)  # Create folder if it doesn't exist
    filename = "recording.wav"
    filepath = os.path.join(recordings_folder, filename)

    # Save to file
    write(filepath, samplerate, audio)
    print(f"✅ Saved to: {filepath}")

    return filepath

if __name__ == "__main__":
    record_audio(duration_sec=10)
