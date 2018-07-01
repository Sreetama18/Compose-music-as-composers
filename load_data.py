from pydub import AudioSegment
import os


def check_and_create_folder(root_dir, intermediate_file):
    target_folder = os.path.join(root_dir, intermediate_file)
    print(target_folder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)


def get_output_filename(input_filename, type, save_folder):
    output_file = input_filename.split('\\')
    output_file_name = output_file[len(output_file) - 1]
    output_file_name = output_file_name[:-4] + type
    folder_name = output_file[len(output_file) - 2]
    check_and_create_folder('..\\' + save_folder, folder_name)
    output_file = os.path.join('..\\', save_folder, folder_name, output_file_name)
    print(output_file)
    return output_file


def convert_to_wav(mp3_filename):
    output_filename = get_output_filename(mp3_filename, '.wav', 'wavfiles')
    sound = AudioSegment.from_mp3(mp3_filename)
    sound.export(output_filename)


def recursively_walk_through_directory(root_dir):

    for root, directories, files in os.walk(root_dir):
        for directory in directories:
            for file in os.listdir(os.path.join(root, directory)):
                mp3_file = os.path.join(root, directory, file)
                print(mp3_file)
                convert_to_wav(mp3_file)


if __name__ == '__main__':
    recursively_walk_through_directory('../Dataset')