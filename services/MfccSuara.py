import librosa
import numpy as np

def extract_mfcc(audio_data, sr, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    return mfccs
