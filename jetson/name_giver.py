import os, sys
import argparse
import random, string
import uuid
import glob

class NameGiver():
    def __init__(self, start_index):
        self.index = start_index
        self.renamed_files = 0

    def generate_unique_name(self, deployment = False):
        ''' Generate a unique name with (for convenience in cleaning data for training)
         or without index before name'''

        unique_name = uuid.uuid4()

        if deployment:
            given_name = unique_name
        else:
            indexeded_name = self.index
            given_name = str(indexeded_name) + '__' + str(unique_name)
            self.index += 1

        return given_name

    def save_with_new_name(self, filename, class_label=False):
        ''' Save image in given output directory under class name
        or rename file in place'''

        new_name = self.generate_unique_name() + '.' + self.get_file_extension(filename)
        if args.save_inplace:
            os.rename(filename, new_name)
        else:
            if class_label:
                save_dir = args.output_directory + class_label + '/' + new_name
            else:
                save_dir = args.output_directory + new_name
            os.rename(filename, save_dir)

        self.renamed_files += 1

    def get_file_extension(self, filename):
        ''' Get the extension of the file.
        Expected extensions - ['.jpg', '.png', '.jpeg'] '''

        return filename.split('.')[-1]


    def get_class_label(self, filename):
        ''' Get class label [Goggles / Glasses / Neither] that the image belongs to '''

        class_label = ''
        if '/Goggles/' in filename or '/goggles/' in filename:
            class_label = 'Goggles'
        elif '/Glasses/' in filename or '/glasses/' in filename:
            class_label = 'Glasses'
        else:
            class_label = 'Neither'

        return class_label

def get_all_files(input_directory, extensions):
    ''' Return all filenames to be renamed (to a unique name) in a list '''

    filenames = []

    # Set the directory you want to start from
    rootDir = input_directory
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for filename in fileList:
            ext = '.' + filename.split('.')[-1]
            if ext in extensions:
                filenames.append(dirName + '/' + filename)

    return filenames

def main():
    #Create output_directory if it doesn't exist
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    name_giver = NameGiver(start_index=0 )
    filenames = get_all_files(args.input_directory, ['.jpg', '.png'])
    print (len(filenames))
    for file in filenames:
        class_label = name_giver.get_class_label(file)
        name_giver.save_with_new_name(file, class_label)

    print (f"{name_giver.renamed_files} files renamed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Name Changer")
    parser.add_argument('--input_directory', '-i', type=str, required=True, help="Input directory containing files")
    parser.add_argument('--output_directory', '-o', default='new_name_directory/', action='store_true', help="Output directory to save files")
    parser.add_argument('--save_inplace', '-c', default=False, action='store_true', help="Rename files in input directory")
    args = parser.parse_args()

    main()

    exit()
