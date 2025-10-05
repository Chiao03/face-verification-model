import os
import argparse
import sys
from  preprocessing import apply_noise_and_occlusion

def main():
    parser = argparse.ArgumentParser(description="Apply noise and occlusion to images")
    
    parser.add_argument('--input_folder', default='../data/alignedfaces', type=str, help="Path to the folder containing aligned images.")
    parser.add_argument('--output_folder', default='../data/distortedfaces', type=str, help="Path to the folder where distorted images will be saved.")
    
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        sys.exit(1)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    apply_noise_and_occlusion(input_folder, output_folder)
    print(f"Process completed. Images saved to: {output_folder}")

if __name__ == "__main__":
    main()