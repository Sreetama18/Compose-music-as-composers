import os
import sys
import importlib
import load_data
from chord_and_beat.beat_detection import beat_detection


def start_work(root_dir):

    for root, directories, files in os.walk(root_dir):
        for directory in directories:
            for file in os.listdir(os.path.join(root, directory)):
                wav_file = os.path.join(root, directory, file)
                print(wav_file)
                beat_detection(wav_file)


if __name__ == '__main__':
    start_work('..\\wavfiles')