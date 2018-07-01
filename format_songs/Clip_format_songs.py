from pydub import AudioSegment
import os
import math

clip_size = 30 * 1000


def clip_songs(sound_file):

    sound = AudioSegment.from_wav(sound_file)
    song_length = len(sound)
    clip_segments = math.ceil(song_length / clip_size)
    new_len = clip_segments * clip_size
    if song_length < new_len:
        duration = new_len - song_length
        sound = sound + add_silence(duration)
    sound.export(sound_file, format="wav")

    clips = []
    new_len = 0
    for i in range(0, clip_segments):
        startTime = i * clip_size
        endTime = (i+1) * clip_size
        clip = sound[startTime:endTime]
        clips.append(clip)


def add_silence(duration):

    print("duration = ", duration)
    silence = AudioSegment.silent(duration=duration)
    return silence


def main():

    print('Enter the root path')
    path = input()

    for root, dirs, files in os.walk(path):
        for directory in dirs:
            for sound_file in os.listdir(path + directory):
                print(sound_file)
                clip_songs(path + directory + '\\' + sound_file)


if __name__ == "__main__":

    main()