from scipy.io import wavfile
import numpy as np
import librosa
from matplotlib import pyplot as plt


def read_wav_data(sound_file):

    signal, sampling_rate = librosa.load(sound_file)
    return signal, sampling_rate
