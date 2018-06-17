import music21


def find_key(sound_file):

    score = music21.converter.parse(sound_file)
    print('score', score)
    key = score.analyze('key')
    print(key.tonic.name, key.mode)


def main():

    # Input sound file in wav format
    print('Enter midi file path')
    sound_file = input()
    find_key(sound_file)


if __name__ == "__main__":
    main()
