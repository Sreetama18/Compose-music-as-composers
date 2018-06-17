import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.fftpack import fft
from scipy.io import wavfile


def show_signal_wave(sound_file, output_file_signal):

    #Plot the audio signal
    spf = wave.open(sound_file, 'r')
    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()
    Time = np.linspace(0, len(signal)/fs, num=len(signal))

    plt.title('Signal Wave...')
    plt.plot(Time, signal)
    plt.savefig(output_file_signal)


def find_fft(sound_file, output_file_fft):

    fs1, data = wavfile.read(sound_file)
    a = data.T[0]
    print(a)
    b = []
    for ele in np.array(a):
        (ele / 2 ** 8.) * 2 - 1
        b.append(ele)
    c = fft(b)
    d = len(c)/2

    plt.title('FFT of Signal Wave...')
    plt.plot(abs(c[:(d-1)]), 'r')
    plt.savefig(output_file_fft)


def main():

    # Input sound file in wav format
    print('Enter wav file path')
    sound_file = raw_input()
    print('Enter output file path for signal plot')
    output_file_signal = raw_input()
    show_signal_wave(sound_file, output_file_signal)
    print('Enter output file path for FFT plot')
    output_file_fft = raw_input()
    find_fft(sound_file, output_file_fft)


if __name__ == "__main__":
    main()
