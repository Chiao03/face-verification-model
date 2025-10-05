import argparse
import numpy as np
import os
import sys
import cv2
from preprocessing.data_creator import read_pairs_file, create_folds_from_sets
from metric_calculator import accuracy
from nested_cross_validation import nested_cross_validation
from cnn_resnet50 import resnet50
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn import svm


def pairimage2feature(pair):
    """
    Purpose: Converts a pair of images to a feature vector

    Parameters:
    - pair (tuple): A pair of image file paths
    
    Result:
    - image (torch.Tensor): A feature vector representing the pair of images

    Howto:
    - Call this function with a pair of image file paths
    - The function will return a feature vector representing the pair of images
    """
    #Unpack the pair and read images
    file1, file2 = pair  
    image1 = cv2.imread(file1)
    image2 = cv2.imread(file2)
    # Make sure the images are read correctly
    if image1 is None or image2 is None:
        print(f"Error reading one of the images: {file1}, {file2}")
        return None

    # Desired size
    height = 128
    width = 128
    desired_size = (height, width)

    # Resize the images
    image1 = cv2.resize(image1, desired_size)
    image2 = cv2.resize(image2, desired_size)

    # Change shape to (num_channels, height, width)
    image1 = image1.transpose((2, 0, 1))
    image2 = image2.transpose((2, 0, 1))

    # Concatenate along the channel dimension
    image = np.concatenate([image1, image2], axis=0)

    # Convert to PyTorch tensor and normalize
    image = torch.from_numpy(image).float() / 255.0

    return image

def files2X(file_pairs, fnc2vector):
    """
    Purpose: Converts a list of file pairs to a feature matrix

    Parameters:
    - file_pairs (list): List of file pairs
    - fnc2vector (function): Function to convert a pair of images to a feature vector

    Result:
    - X (np.ndarray): Feature matrix representing the file pairs
    """
    X = []
    for file1, file2 in file_pairs:
        vector = fnc2vector((file1, file2))
        if vector is not None:
            X.append(vector)
    return np.array(X)

def train(train_files, fnc2vector, params):
    """
    Purpose: Train a CNN model and SVM model on the training data

    Parameters:
    - train_files (dict): Dictionary of matched and mismatched file pairs
    - fnc2vector (function): Function to convert a pair of images to a feature vector
    - params (dict): Dictionary of hyper-parameters for the CNN model

    Result:
    - cnn_model (nn.Module): Trained CNN model
    - svm_model (svm.SVC): Trained SVM model

    Howto:
    - Call this function with the training data and hyper-parameters
    - The function will train a CNN model and SVM model on the training data
    - The function will return the trained CNN and SVM models
    """
    X_train = files2X(train_files['matched'], fnc2vector)
    y_train = np.ones(len(train_files['matched']))

    X_train_mismatch = files2X(train_files['mismatched'], fnc2vector)
    y_train_mismatch = np.zeros(len(train_files['mismatched']))

    X_train = np.concatenate([X_train, X_train_mismatch], axis=0)
    y_train = np.concatenate([y_train, y_train_mismatch], axis=0)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    # Create a DataLoader for the training data
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Use ResNet50 model
    model = resnet50(params['kernel_size'])

    # Define a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 1
    for epoch in range(epochs):

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    # Remove output layer
    cnn_model = nn.Sequential(*list(model.children())[:-1])

    # Extract features from the training data
    X_train_transformed = extract_features(model, X_train).numpy()

    # Reshape the data to 2D
    X_train_transformed_2d = X_train_transformed.reshape(X_train_transformed.shape[0], -1)

    # Train the SVM
    params['C'] = 0.01
    svm_model = svm.SVC(C=params['C'], kernel='linear')
    svm_model.fit(X_train_transformed_2d, y_train)

    return cnn_model, svm_model

def extract_features(model, X):
    """
    Purpose: Extract features from the CNN model

    Parameters:
    - model (nn.Module): The CNN model
    - X (torch.Tensor): Input data

    Result:
    - outputs (torch.Tensor): Output features from the CNN model
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
    return outputs

def evaluate(cnn_model, svm_model, test_files, fnc2vector, metric_function):
    """
    Purpose: Evaluate the CNN and SVM models on the testing data

    Parameters:
    - cnn_model (nn.Module): The trained CNN model
    - svm_model (svm.SVC): The trained SVM model
    - test_files (dict): Dictionary of matched and mismatched file pairs
    - fnc2vector (function): Function to convert a pair of images to a feature vector
    - metric_function (function): Function to calculate the evaluation metric

    Result:
    - metric (float): Evaluation metric
    - y_test (torch.Tensor): True labels
    - y_pred (np.ndarray): Predicted labels
    - y_score (np.ndarray): Predicted scores

    Howto:
    - Call this function with the trained models and testing data
    - The function will evaluate the models on the testing data
    - The function will return the evaluation metric, true labels, predicted labels, and predicted scores
    """
    X_test = files2X(test_files['matched'], fnc2vector)
    y_test = np.ones(len(test_files['matched']))

    X_test_mismatch = files2X(test_files['mismatched'], fnc2vector)
    y_test_mismatch = np.zeros(len(test_files['mismatched']))

    X_test = np.concatenate([X_test, X_test_mismatch], axis=0)
    y_test = np.concatenate([y_test, y_test_mismatch], axis=0)

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    # Extract features from the testing data using the CNN model
    X_test_transformed = extract_features(cnn_model, X_test).numpy()

    X_test_transformed_2d = X_test_transformed.reshape(X_test_transformed.shape[0], -1)

    # Predict with the SVM
    y_pred = svm_model.predict(X_test_transformed_2d)
    y_score = svm_model.decision_function(X_test_transformed_2d)

    return metric_function(y_test, y_pred), y_test, y_pred, y_score

def main():
    parser = argparse.ArgumentParser(description="Training SVM-CNN model with nested cross validation")
    parser.add_argument('--pairfile', type=str, default='pairs.txt', help="Path to the pairs file containing image pairs. Default: 'pairs.txt'")
    parser.add_argument('--datafolder', type=str, default='data/alignedfaces', help="Folder path containing face images. Default: 'data/alignedfaces'")
    parser.add_argument('--resultfolder', type=str, default='results', help="Folder to store the output results. Default: 'results'")
    parser.add_argument('--resultfile', type=str, default='alignedcnn', help="Filename for storing results (without extension). Default: 'alignedcnn'")

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
    param_grid = [{'kernel_size': 1}, {'kernel_size': 3}, {'kernel_size': 5}]

    result_path = os.path.join(result_folder, result_file)
    nested_cross_validation(folds, train, evaluate, param_grid, accuracy, pairimage2feature, result_path)

if __name__ == "__main__":
    main()
