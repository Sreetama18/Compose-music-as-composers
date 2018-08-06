import librosa
import librosa.display
import matplotlib.pylab as plt
import numpy as np
import music21
import os
from pydub import AudioSegment
import csv
import math

features = []
clip_size = 30 * 1000

def format_converter(path, sound_file):

    sound = AudioSegment.from_mp3(sound_file)
    sound.export(path, format="midi")


def find_pitch(signal, sampling_rate):
    
    freqs = np.fft.rfft(signal)
    auto1 = freqs * np.conj(freqs)
    auto2 = auto1 * np.conj(auto1)
    result = np.fft.irfft(auto2)

    return np.array(result)


def find_key(path, sound_file):

    soundfile = format_converter(path, sound_file)
    score = music21.converter.parse(soundfile)
    key = score.analyze('key')
    return key

def plot_static_beat(tempo, onset_env, sampling_rate):
    # Convert to scalar
    tempo = np.asscalar(tempo)
    # Compute 2-second windowed autocorrelation
    hop_length = 512
    auto_correlation = librosa.autocorrelate(onset_env, 2 * sampling_rate // hop_length)
    freqs = librosa.tempo_frequencies(len(auto_correlation), sr=sampling_rate, hop_length=hop_length)

    # Plot on a BPM axis.  We skip the first (0-lag) bin.
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.semilogx(freqs[1:], librosa.util.normalize(auto_correlation)[1:], label='Onset autocorrelation', basex=2)
    ax.axvline(tempo, 0, 1, color='r', alpha=0.75, linestyle='--', label='Tempo: {:.2f} BPM'.format(tempo))
    ax.grid()
    ax.axis('tight')
    return fig


def plot_dynamic_beat(dtempo, onset_env, sampling_rate):
    hop_length = 512
    fig = plt.figure()
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sampling_rate, hop_length=hop_length)
    librosa.display.specshow(tg, x_axis='time', y_axis='tempo')
    ax = fig.add_subplot(111)
    ax.plot(librosa.frames_to_time(np.arange(len(dtempo))), dtempo, color='w', linewidth=1.5, label='Tempo estimate')

    return fig


def get_onset(signal, sampling_rate):
    return librosa.onset.onset_strength(signal, sr=sampling_rate)


def beat_detection(signal, sampling_rate):
    onset_envelope = get_onset(signal, sampling_rate)
    #tempo = static_tempo_detection(onset_envelope, sampling_rate)
    dynamic_tempo = dynamic_beat_detection(onset_envelope, sampling_rate)

    return np.array(dynamic_tempo)


def static_tempo_detection(onset_env, sampling_rate):
    return librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)


def dynamic_beat_detection(onset_envelope, sampling_rate):
    return librosa.beat.tempo(onset_envelope=onset_envelope, sr=sampling_rate, aggregate=None)


def clip_songs(sound_file):

    sound = AudioSegment.from_wav(sound_file)
    song_length = len(sound)
    clip_segments = math.ceil(song_length / clip_size)
    new_len = clip_segments * clip_size
    if song_length < new_len:
        duration = new_len - song_length
        sound = sound + add_silence(duration)

    clips = []
    new_len = 0
    for i in range(0, clip_segments):
        startTime = i * clip_size
        endTime = (i+1) * clip_size
        clip = sound[startTime:endTime]
        clips.append(clip)
        clips.append(clip)

    return clips


def add_silence(duration):

    #print("duration = ", duration)
    silence = AudioSegment.silent(duration=duration)
    return silence


def output_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():

    max_len = 0
    count = 0
    print('Enter the wav dataset path')
    path = input()
    for root, dirs, files in os.walk(path):
        for name in files:
            print("name = ", name)
            filename = path + name
            clips = clip_songs(filename)
            features = []
            count = count + 1
            for clip in clips:
                clip.export("d:\\temp.wav", format="wav")
                signal, sampling_rate = librosa.load("d:\\temp.wav")
                pitch = find_pitch(signal, sampling_rate)
                beat = beat_detection(signal, sampling_rate)
                pitch_len = len(pitch)
                beat_len = len(beat)
                curr_max_len = max(pitch_len, beat_len)
                if curr_max_len > max_len:
                    max_len = curr_max_len
                    diff_len = curr_max_len - max_len
                    for list in features:
                        list.extend([0] * diff_len)
                    else:
                        if pitch_len < max_len:
                            diff = max_len - pitch_len
                            result_pitch = np.zeros(max_len)
                            result_pitch[:pitch.shape[0]] = pitch
                        if beat_len < max_len:
                            diff = max_len - beat_len
                            result_beat = np.zeros(max_len)
                            result_beat[:beat.shape[0]] = beat

                    features.append(pitch)
                    features.append(beat)

                output_path = path + "Data_csv\\"
                output_dir(output_path)
                myfile = open(output_path + "song{0}.csv".format(count), 'a')
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=',')
                wr.writerow(features)
                myfile.close()


if __name__ == "__main__":
    main()