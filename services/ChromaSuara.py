import librosa
import numpy as np

def extract_chroma(audio_data, sr):
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    return chroma
