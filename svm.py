import argparse
import os
import sys
import pandas as pd
import numpy as np
import cv2
from sklearn import svm
from preprocessing.data_creator import read_pairs_file, create_folds_from_sets
from metric_calculator import accuracy
from nested_cross_validation import nested_cross_validation



def hog_features(image):
    """
    Purpose:
    Extract Histogram of Oriented Gradients (HOG) features from the given image.

    Parameters:
    - image (numpy array): Input image, either in grayscale or in color format (BGR).

    Result:
    Returns a 1D feature vector containing HOG features extracted from the image.

    Howto:
    - If the input image is in color (3 channels), it converts it to grayscale.
    - Resizes the image to a fixed size of 64x128 pixels (standard size for HOG).
    - Computes the HOG features using OpenCV's HOGDescriptor.
    - Flattens and returns the feature vector as a 1D array.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 64x128 for HOG
    image = cv2.resize(image, (64, 128))
    hog = cv2.HOGDescriptor()
    feature_vector = hog.compute(image)
    return feature_vector.flatten()


def pairimage2feature(pair):
    """
    Purpose:
    Compute the feature difference between a pair of images using HOG features.

    Parameters:
    - pair (tuple of str): A tuple containing the file paths of two images to be compared.

    Result:
    Returns a feature vector representing the absolute difference between the HOG feature vectors of the two images.
    If either image cannot be read, returns `None` and prints an error message.

    Howto:
    - Reads both images using OpenCV.
    - Converts each image into HOG feature vectors using the `hog_features` function.
    - Computes the absolute difference between the two feature vectors.
    - Returns the resulting feature vector.
    """
    file1, file2 = pair  # file1 and file2 are file paths

    # Read the images using OpenCV
    image1 = cv2.imread(file1)
    image2 = cv2.imread(file2)

    # Make sure the images are read correctly
    if image1 is None or image2 is None:
        print(f"Error reading one of the images: {file1}, {file2}")
        return None

    feature1 = hog_features(image1)
    feature2 = hog_features(image2)

    vector = np.abs(feature1 - feature2)
    return vector

def files2X(file_pairs, fnc2vector):
    """
    Purpose:
    Convert a list of file pairs into a feature matrix using a specified feature extraction function.

    Parameters:
    - file_pairs (list of tuples): A list of tuples, where each tuple contains the file paths of two images to be processed.
    - fnc2vector (function): A function that takes a pair of file paths as input and returns a feature vector.

    Result:
    Returns a NumPy array containing the feature vectors for each pair of images.
    """
    X = []
    for file1, file2 in file_pairs:
        vector = fnc2vector((file1, file2))
        if vector is not None:
            X.append(vector)
    return np.array(X)


def train(train_files, fnc2vector, params):
    """
    Purpose:
    Train a Support Vector Machine (SVM) model using the given training files and parameters.

    Parameters:
    - train_files (dict): A dictionary containing two keys: 'matched' and 'mismatched'.
                          Each key corresponds to a list of file paths for matched and mismatched image pairs, respectively.
    - fnc2vector (function): A function that converts a pair of image file paths into a feature vector.
    - params (dict): A dictionary containing hyperparameters for the SVM model, specifically the regularization parameter 'C'.

    Result:
    Returns a trained SVM model.

    Howto:
    - Extracts feature vectors for all matched image pairs using `files2X` and assigns label 1.
    - Extracts feature vectors for all mismatched image pairs and assigns label 0.
    - Concatenates both matched and mismatched feature vectors and labels to form the training dataset.
    - Trains the SVM model using the combined training dataset.
    - Returns the trained SVM model.
    """
    X_train = files2X(train_files['matched'], fnc2vector)
    y_train = np.ones(len(train_files['matched']))

    X_train_mismatch = files2X(train_files['mismatched'], fnc2vector)
    y_train_mismatch = np.zeros(len(train_files['mismatched']))

    X_train = np.concatenate([X_train, X_train_mismatch], axis=0)
    y_train = np.concatenate([y_train, y_train_mismatch], axis=0)

    model = svm.SVC(C=params['C'], kernel='linear')
    model.fit(X_train, y_train)
    return model


def evaluate(model, test_files, fnc2vector, metric_function):
    """
    Purpose:
    Evaluate the performance of a trained model on a test set using a specified evaluation metric.

    Parameters:
    - model (object): A trained machine learning model to be evaluated.
    - test_files (dict): A dictionary containing two keys: 'matched' and 'mismatched'.
                         Each key corresponds to a list of file paths for matched and mismatched image pairs, respectively.
    - fnc2vector (function): A function that converts a pair of image file paths into a feature vector.
    - metric_function (function): A function to compute the evaluation metric based on true and predicted labels.

    Result:
    Returns a tuple containing:
    - Evaluation metric value (e.g., accuracy) computed using `metric_function`.
    - y_test (numpy array): True labels for the test set.
    - y_pred (numpy array): Predicted labels for the test set.
    - y_score (numpy array): Confidence scores or decision function values for each prediction.

    """
    X_test = files2X(test_files['matched'], fnc2vector)
    y_test = np.ones(len(test_files['matched']))

    X_test_mismatch = files2X(test_files['mismatched'], fnc2vector)
    y_test_mismatch = np.zeros(len(test_files['mismatched']))

    X_test = np.concatenate([X_test, X_test_mismatch], axis=0)
    y_test = np.concatenate([y_test, y_test_mismatch], axis=0)

    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)
    return metric_function(y_test, y_pred), y_test, y_pred, y_score


def main():
    parser = argparse.ArgumentParser(description="Training svm model with nested cross validation")
    parser.add_argument('--pairfile', type=str, default='pairs.txt', help="Path to the pairs file containing image pairs. Default: 'pairs.txt'")
    parser.add_argument('--datafolder', type=str, default='data/alignedfaces', help="Folder path containing face images. Default: 'data/alignedfaces'")
    parser.add_argument('--resultfolder', type=str, default='results', help="Folder to store the output results. Default: 'results'")
    parser.add_argument('--resultfile', type=str, default='svm_aligned', help="Filename for storing results (without extension). Default: 'alignedsvm'")

    args = parser.parse_args()
    pair_file = args.pairfile
    data_folder = args.datafolder
    result_folder = args.resultfolder
    result_file = args.resultfile

    if not os.path.exists(data_folder):
        print(f"Error: The data folder '{data_folder}' does not exist.")
        sys.exit(1)
    
    if not os.path.isfile(pair_file):
        print(f"Error: The pair file '{pair_file}' does not exist.")
        sys.exit(1) 

    os.makedirs(result_folder, exist_ok=True)

    matched_sets, mismatched_sets = read_pairs_file(file_path=pair_file, data_folder=data_folder)
    # Create 20 folds
    folds = create_folds_from_sets(matched_sets, mismatched_sets)
    param_grid = [{'C': 0.001}, {'C': 0.01}, {'C': 0.1}]

    result_path = os.path.join(result_folder, result_file)
    nested_cross_validation(folds, train, evaluate, param_grid, accuracy, pairimage2feature, result_path)

if __name__ == "__main__":
    main()