import argparse
import os
import sys
from preprocessing import align_faces_in_folder

def main():
    parser = argparse.ArgumentParser(description="Align faces in images and save them to a specified folder.")

    parser.add_argument('--input_folder', type=str, default='../lfw', help="Path to the folder containing input images. Default: '../lfw'")
    parser.add_argument('--output_folder', type=str, default='../data/alignedfaces', help="Path to the folder where aligned images will be saved. Default: '../data/alignedfaces'")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        sys.exit(1)  

    os.makedirs(output_folder, exist_ok=True)
    align_faces_in_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
    

