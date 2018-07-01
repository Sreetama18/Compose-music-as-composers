import librosa
import read_wav_data
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import load_data


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
    # fig.xlabel('Tempo (BPM)')
    ax.grid()
    # ax.title('Static tempo estimation')
    ax.axis('tight')
    return fig


def plot_dynamic_beat(dtempo, onset_env, sampling_rate):
    hop_length = 512
    fig = plt.figure()
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sampling_rate, hop_length=hop_length)
    librosa.display.specshow(tg, x_axis='time', y_axis='tempo')
    ax = fig.add_subplot(111)
    ax.plot(librosa.frames_to_time(np.arange(len(dtempo))), dtempo, color='w', linewidth=1.5, label='Tempo estimate')
    # fig.title('Dynamic tempo estimation')
    # fig.legend(frameon=True, framealpha=0.75)
    return fig


def get_onset(signal, sampling_rate):
    return librosa.onset.onset_strength(signal, sr=sampling_rate)


def beat_detection(filename):
    signal, sampling_rate = read_wav_data.read_wav_data(filename)
    onset_envelope = get_onset(signal, sampling_rate)
    tempo = static_tempo_detection(onset_envelope, sampling_rate)
    dynamic_tempo = dynamic_beat_detection(onset_envelope, sampling_rate)
    static_fig = plot_static_beat(tempo, onset_envelope, sampling_rate)
    dynamic_fig = plot_dynamic_beat(dynamic_tempo, onset_envelope, sampling_rate)

    static_fig.savefig(load_data.get_output_filename(filename, '_s.png', 'static_beat'))
    dynamic_fig.savefig(load_data.get_output_filename(filename, '_d.png', 'dynamic_beat'))


def static_tempo_detection(onset_env, sampling_rate):
    return librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)


def dynamic_beat_detection(onset_envelope, sampling_rate):
    return librosa.beat.tempo(onset_envelope=onset_envelope, sr=sampling_rate, aggregate=None)