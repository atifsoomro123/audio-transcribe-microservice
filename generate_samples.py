from gtts import gTTS
import os

# Create folder for samples
os.makedirs("sample_audio", exist_ok=True)

samples = {
    "clean_speech": "Hello world, this is a clear audio sample for testing transcription.",
    "noisy_background": "This audio simulates background noise and will be used for testing noise suppression.",
    "multi_speaker": "Speaker one says hello. Speaker two replies good morning!"
}

for name, text in samples.items():
    tts = gTTS(text=text, lang="en")
    wav_path = f"sample_audio/{name}.wav"
    tts.save(wav_path)
    print(f"✅ Created {wav_path}")

print("\nAll sample WAV files are ready in ./sample_audio/")
