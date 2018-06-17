import librosa
import librosa.display
import matplotlib.pylab as plt
import numpy as np


def find_pitch(sound_file, output_file):

    y, sr = librosa.load(sound_file)
    freqs = np.fft.rfft(y)
    auto1 = freqs * np.conj(freqs)
    auto2 = auto1 * np.conj(auto1)
    result = np.fft.irfft(auto2)
    print(result)
    plt.title('Pitch of Signal Wave')
    plt.plot(result)
    plt.savefig(output_file)


def main():

    # Input sound file in wav format
    print('Enter wav file path')
    sound_file = raw_input()
    print('Enter output file path')
    output_file = raw_input()
    find_pitch(sound_file, output_file)


if __name__ == "__main__":

    main()
