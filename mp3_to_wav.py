import os


def convert_mp3_to_wav(root_dir) :

    os.chdir(root_dir)
    for filename in os.listdir(root_dir):
        # SOX
        os.system("sox " + str(filename) + " " + str(filename[:-4]) + ".wav")

    os.system("rm *.mp3")


def main():

    # Input sound file in wav format
    print('Enter the directory containing mp3 files')
    root_dir = input()
    convert_mp3_to_wav(root_dir)


if __name__ == "__main__":

    main()