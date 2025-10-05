import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from matplotlib import pyplot as plt
import cv2
import numpy as np
from IPython.utils import io
from mtcnn import MTCNN
import random

def align_face_mtcnn(image_path):
    """
    Purpose:
    ----------
    Detect and align a face in an image using MTCNN by aligning the eyes horizontally.
    This function helps in preprocessing face images for face recognition tasks.

    Parameters:
    ----------
    - image_path: The file path to the input image containing the face to be aligned.

    Output:
    ----------
    - Returns a cropped and aligned image (numpy array) if a face is detected.
    - If no face is detected, prints "No face detected" and returns `None`.

    Note:
    ----------
    This function assumes that the first face detected is the primary face of interest.
    """
    with io.capture_output() as captured:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detector = MTCNN()
        results = detector.detect_faces(image_rgb)

    if len(results) > 0:
        face = results[0]
        keypoints = face['keypoints']

        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        M = cv2.getRotationMatrix2D(eyes_center, angle, 1)

        aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        x, y, w, h = face['box']
        aligned_face_cropped = aligned_face[y:y+h, x:x+w]
        
        return aligned_face_cropped
    else:
        print("No face detected.")


def add_noise(image, noise_factor=0.05):
    """
    Purpose:
    ----------
    Add Gaussian noise to an image for data augmentation or robustness testing.
    This function is typically used in scenarios where the effect of noise on image recognition algorithms is evaluated.

    Parameters:
    ----------
    - image: Input image as a numpy array (assumed to be in 8-bit unsigned integer format).      
    - noise_factor: A floating-point value indicating the intensity of the noise to be added. 
      Default value is 0.05, meaning the standard deviation of the noise is 5% of the pixel value range.

    Output:
    ----------
    - Returns a noisy version of the input image as a numpy array in 8-bit unsigned integer format.
    """
    image = image.astype(np.float32) / 255.0
    mean = 0.0
    std = 1.0
    gaussian_noise = np.random.normal(mean, std, image.shape)
    gaussian_noise = gaussian_noise * noise_factor
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image

def add_occlusion(image, min_size=10, max_size=50, occlusion_color=(0, 0, 0)):
    """
    Purpose:
    ----------
    Add a random occlusion to an image using different geometric shapes (rectangle, square, or triangle).
    This function is typically used for testing the robustness of image recognition algorithms under partial occlusion.

    Parameters:
    ----------
    - image: Input image as a numpy array.
    - min_size: The minimum size of the occlusion shape. Default is 10 pixels.      
    - max_size: The maximum size of the occlusion shape. Default is 50 pixels.      
    - occlusion_color: A tuple representing the RGB color of the occlusion shape. Default is black (0, 0, 0).

    Output:
    ----------
    - Returns a copy of the input image with a randomly placed occlusion.

    Note:
    ----------
    The occlusion size and position are randomly selected for each call, allowing for diverse occluded images.
    """
    occluded_image = image.copy()

    size = random.randint(min_size, max_size)

    x = random.randint(0, max(0, image.shape[1] - size))  
    y = random.randint(0, max(0, image.shape[0] - size))  

    shape_type = random.choice(['rectangle', 'square', 'triangle'])

    if shape_type == 'rectangle':
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])
        cv2.rectangle(occluded_image, (x, y), (x_end, y_end), occlusion_color, -1)

    elif shape_type == 'square':
        x_end = min(x + size, image.shape[1])
        y_end = min(y + size, image.shape[0])
        cv2.rectangle(occluded_image, (x, y), (x_end, y_end), occlusion_color, -1)

    elif shape_type == 'triangle':
        vertices = np.array([
            [x, y],  
            [x + size, y],  
            [x + size // 2, y + size] 
        ])
        cv2.drawContours(occluded_image, [vertices], 0, occlusion_color, -1)

    return occluded_image

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def lateral(image1, image2):
    """
    Purpose: Combine two images side by side (horizontally) into a single image.
    Parameters: 2 image
    Output: Returns a single image with `image1` and `image2` placed side by side.
    """
    target_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))  
    image1_resized = resize_image(image1, target_size)
    image2_resized = resize_image(image2, target_size)
    return np.hstack((image1_resized, image2_resized))


def layered(image1, image2):
    """
    Purpose: Combine two images into a layered RGB image
    Parameters: 2 image
    Output: Returns an RGB image with R, G, and B channels constructed from the two input images.
    """
    target_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))  
    image1_resized = resize_image(image1, target_size)
    image2_resized = resize_image(image2, target_size)
    
    gray1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)
    
    layered = np.dstack((gray1, gray2, (gray1 + gray2) // 2))
    return layered


def stack(image1, image2):
    """
    Purpose: Combine two images vertically into a single image.
    Parameters: 2 image
    Output: Returns a single image with `image1` stacked on top of `image2`.
    """
    target_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0])) 
    image1_resized = resize_image(image1, target_size)
    image2_resized = resize_image(image2, target_size)
    return np.vstack((image1_resized, image2_resized))


def hog_features(image):   
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 128))
    hog = cv2.HOGDescriptor()
    feature_vector = hog.compute(image)
    return feature_vector.flatten()

def align_faces_in_folder(input_folder, output_folder):
    """
    Purpose:
    ----------
    Detect and align faces in all images within a specified input folder using MTCNN, and save the aligned faces to an output folder.

    Parameters:
    ----------
    - input_folder: The path to the folder containing input images.

    - output_folder: The path to the folder where aligned images will be saved.

    How To:
    ----------
    1. Create the output folder if it does not exist.
    2. Traverse through each subdirectory and image file in the `input_folder`.
    3. For each image:
       a. Read the image and detect faces using MTCNN.
       b. If a face is detected, extract the facial keypoints and calculate the rotation angle.
       c. Rotate and crop the image to align the face.
       d. Save the aligned face image to the corresponding subdirectory in the `output_folder`.
    4. If no face is detected, skip the image and print a warning.

    Output:
    ----------
    - Saves the aligned face images to the specified output folder, preserving the subdirectory structure.
    - Prints the status of each processed image.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    detector = MTCNN()

    for subdir, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  
                image_path = os.path.join(subdir, filename)
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to read image {filename}. Skipping.")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(image_rgb)

                if len(results) > 0:
                    face = results[0]
                    keypoints = face['keypoints']

                    left_eye = keypoints['left_eye']
                    right_eye = keypoints['right_eye']

                    delta_x = right_eye[0] - left_eye[0]
                    delta_y = right_eye[1] - left_eye[1]
                    angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

                    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

                    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)

                    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)


                    x, y, w, h = face['box']
                    aligned_face_cropped = aligned_face[y:y+h, x:x+w]
                    
                    relative_path = os.path.relpath(subdir, input_folder) 
                    output_subdir = os.path.join(output_folder, relative_path)

                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    output_path = os.path.join(output_subdir, filename)
                    cv2.imwrite(output_path, aligned_face_cropped)
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"No face detected in image: {filename}. Skipping.")


def apply_noise_and_occlusion(input_folder, output_folder):
    """
    Applies noise, occlusion, or both to images in the input folder and saves them to the output folder.
    The operation is chosen randomly for each image with equal probability.
    
    Parameters:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the processed images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  
                image_path = os.path.join(subdir, filename)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to read image {filename}. Skipping.")
                    continue

                operation = random.choice(['noise', 'occlusion', 'both'])

                if operation == 'noise':
                    processed_image = add_noise(image)

                elif operation == 'occlusion':
                    processed_image = add_occlusion(image)

                else: 
                    noisy_image = add_noise(image)
                    processed_image = add_occlusion(noisy_image)

                relative_path = os.path.relpath(subdir, input_folder) 
                output_subdir = os.path.join(output_folder, relative_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_path = os.path.join(output_subdir, filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved with {operation}: {output_path}")