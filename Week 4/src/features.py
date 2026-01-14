import librosa
import numpy as np

def extract_mfcc(path, n_mfcc=13, max_len=100):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T

    if len(mfcc) > max_len:
        mfcc = mfcc[:max_len]
    else:
        mfcc = np.pad(mfcc, ((0, max_len-len(mfcc)), (0,0)))

    return mfcc