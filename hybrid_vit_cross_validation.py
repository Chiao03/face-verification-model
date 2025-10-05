import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from hybrid_vit import HybridViT
import sys
sys.path.append('./preprocessing')
from data_creator import read_pairs_file, create_folds_from_sets, bootstrap_632_sampling
from torch.utils.data import Dataset
from PIL import Image
from hybrid_vit_transform_data import transform_data
import random
import time
from torch import optim
import csv
import os
import argparse


class FaceDataset(Dataset):
    '''
    Description: A custom dataset class for face images
    '''
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        
        # Load the image
        image = Image.open(image_path)  # Convert to grayscale if needed

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label
    
def validate(model, validation_pairs, transforms):
    '''
    Description: Validate the model on the validation set

    Parameters:
    - model (HybridViT): The model to validate
    - validation_pairs (list): A list of tuples containing the image pairs and their labels
    - transforms (torchvision.transforms): The transformations to apply to the images

    Returns:
    - accuracy (float): The accuracy of the model on the validation set
    '''
    model.eval()
    cnt = 0
    matched_similarity = []
    mismatched_similarity = []
    n = len(validation_pairs)
    correct = 0
    accuracies = []
    true_labels = []
    predicted_labels = []
    all_similarities = []

    for pair in validation_pairs:
        print(f'Processing pair {cnt + 1}/{n}', end='\r')
        img1 = Image.open(pair[0]).convert('RGB')
        img1_tensor = transform(img1).unsqueeze(0)
        img2 = Image.open(pair[1]).convert('RGB')
        img2_tensor = transform(img2).unsqueeze(0)
        # Move the model and tensor to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        # Perform inference
        with torch.no_grad(): 
            embeddings1 = model(img1_tensor)
            embeddings2 = model(img2_tensor)

        # Cosine similarity between the embeddings
        similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)
        true_labels.append(pair[2])
        all_similarities.append(similarity.item())
        # Set the threshold for cosine similarity
        if similarity > 0.96:
            predicted_labels.append(1)
            if pair[2] == 1:
                correct += 1
        else:
            predicted_labels.append(0)
            if pair[2] == 0:
                correct += 1

        if pair[2] == 1:
            matched_similarity.append(similarity.item())
        else:
            mismatched_similarity.append(similarity.item())

        cnt += 1
        accuracies.append(correct / n)

    if (len(matched_similarity) == 0):
        print('No matched pairs')
    else:
        print(f'Matched similarity: {sum(matched_similarity) / len(matched_similarity)}')
        print(f'Median: {sorted(matched_similarity)[len(matched_similarity) // 2]}, Min: {min(matched_similarity)}, Max: {max(matched_similarity)}')
    
    if (len(mismatched_similarity) == 0):
        print('No mismatched pairs')
    else:
        print(f'Mismatched similarity: {sum(mismatched_similarity) / len(mismatched_similarity)}')
        print(f'Median: {sorted(mismatched_similarity)[len(mismatched_similarity) // 2]}, Min: {min(mismatched_similarity)}, Max: {max(mismatched_similarity)}')
    print(f'Accuracy: {correct / n}')
    return correct / n, true_labels, predicted_labels, all_similarities
    
def train(train_loader, epochs, learning_rate, depth):
    '''
    Description: Train the model on the training set

    Parameters:
    - train_loader (DataLoader): The DataLoader object for the training set
    - epochs (int): The number of epochs to train the model
    - learning_rate (float): The learning rate for the optimizer
    - depth (int): The depth of the model

    Returns:
    - model (HybridViT): The trained model
    '''

    model = HybridViT(
        image_size=128,
        patch_size=16,
        dim=512,
        depth=depth,
        heads=6,
        mlp_dim=2048,
        num_class=5749,
        channels=1,
        GPU_ID=None # Specify GPU ID if available
    )

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Typically use cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Start counting time for each epoch
        epoch_start_time = time.time()
        
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, labels)

            # Compute the loss
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}', end='\r')
                running_loss = 0.0

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch+1}/{epochs}] completed in {epoch_duration:.2f} seconds')
    
    return model

def cross_validation_with_bootstrap(folds, transforms, bootstrap_iterations, epochs, learning_rate, batch_size, depth_values, filepath):
    '''
    Description: Perform cross-validation with nested bootstrap sampling

    Parameters:
    - folds (list): A list of dictionaries containing the matched and mismatched pairs for each fold
    - transforms (torchvision.transforms): The transformations to apply to the images
    - bootstrap_iterations (int): The number of bootstrap iterations
    - epochs (int): The number of epochs to train the model
    - learning_rate (float): The learning rate for the optimizer
    - batch_size (int): The batch size for training
    - depth_values (list): A list of hyperparameter values for the depth of the model
    - filepath (str): The path to store the results
    '''

    with open(filepath + '.csv', 'w', newline='') as best_params_file:
        best_params_writer = csv.writer(best_params_file)
        best_params_writer.writerow(["Outer Fold", "Best Params", "Inner Accuracy", "Outer Accuracy"])

        with open(filepath + '_labels.csv', 'w', newline='') as labels_file:
            labels_writer = csv.writer(labels_file)
            labels_writer.writerow(["Outer Fold", "True Labels", "Predicted Labels", "Predicted Scores"])

            fold_accuracies = []

            for fold_idx, test_fold in enumerate(folds):
                # if (fold_idx != 0):
                #     continue
                print(f'Fold {fold_idx + 1}/{len(folds)}')
                train_folds = [folds[j] for j in range(len(folds)) if j != fold_idx]
                train_files = prepare_file_names_from_folds(train_folds)
                best_params = None
                best_accuracy = -1
                for depth in depth_values:
                    # if (depth != 2):
                    #     continue
                    print(f'Tuning depth = {depth}')
                    average_bootstrap_accuracy = 0
                    for iteration in range(bootstrap_iterations):
                        # if (iteration != 0):
                        #     continue
                        print(f'Iteration {iteration + 1}/{bootstrap_iterations}')
                        train_sample, validation_sample = bootstrap_632_sampling(
                            train_files['matched'], train_files['mismatched'], random_state=iteration
                        )
                        validation_pairs = []
                        for key in validation_sample.keys():
                            if (key == 'matched'):
                                for pair in validation_sample[key]:
                                    validation_pairs.append((pair[0], pair[1], 1))
                            else:
                                for pair in validation_sample[key]:
                                    validation_pairs.append((pair[0], pair[1], 0))
                        transformed_train = transform_data(train_sample)
                        train_dataset = FaceDataset(data=transformed_train, transform=transform)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

                        model = train(train_loader, epochs, learning_rate, depth)

                        # validation accuracy for this bootstrap iteration
                        print(f'Validating model with depth = {depth}')
                        bootstrap_iteration_accuracy, _, _, _ = validate(model, validation_pairs, transforms)
                        average_bootstrap_accuracy += bootstrap_iteration_accuracy

                    average_bootstrap_accuracy /= bootstrap_iterations
                    if (average_bootstrap_accuracy > best_accuracy):
                        best_accuracy = average_bootstrap_accuracy
                        best_params = depth
                print(f'Best depth for fold {fold_idx + 1}: {best_params} with average accuracy: {best_accuracy}')

                # train the model with the best depth and test it
                transformed_train = transform_data(train_files)
                train_dataset = FaceDataset(data=transformed_train, transform=transform)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                print('Training model with best depth')
                model = train(train_loader, epochs, learning_rate, best_params)
                test_pairs = []
                for key in validation_sample.keys():
                    if (key == 'matched'):
                        for pair in test_fold[key]:
                            test_pairs.append((pair[0], pair[1], 1))
                    else:
                        for pair in test_fold[key]:
                            test_pairs.append((pair[0], pair[1], 0))
                test_accuracy, true_labels, predicted_labels, all_similarities = validate(model, test_pairs, transforms)
                print(f'Test accuracy for fold {fold_idx + 1}: {test_accuracy}')
                fold_accuracies.append(test_accuracy)
                best_params_writer.writerow([fold_idx + 1, best_params, best_accuracy, test_accuracy])

                labels_writer.writerow([
                    fold_idx + 1, 
                    ','.join(map(str, true_labels)), 
                    ','.join(map(str, predicted_labels)),
                    ','.join(map(str, all_similarities))
                ])
            
            average_fold_accuracy = sum(fold_accuracies) / len(fold_accuracies)

            print(f'Average Fold Testing Accuracy: {average_fold_accuracy:.4f}')



def prepare_file_names_from_folds(folds):
    '''
    Description: Prepare the file names for training from the folds

    Parameters:
    - folds (list): A list of dictionaries containing the matched and mismatched pairs for each fold

    Returns:
    - train_files (dict): A dictionary containing matched and mismatched file names for training
    '''
    train_files = {'matched': [], 'mismatched': []}
    for fold in folds:
        train_files['matched'].extend(fold['matched'])      
        train_files['mismatched'].extend(fold['mismatched']) 

    return train_files

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training Hybrid ViT model with nested cross validation")
    parser.add_argument('--pairfile', type=str, default='pairs.txt', help="Path to the pairs file containing image pairs. Default: 'pairs.txt'")
    parser.add_argument('--datafolder', type=str, default='data/alignedfaces', help="Folder path containing face images. Default: 'data/alignedfaces'")
    parser.add_argument('--resultfolder', type=str, default='results', help="Folder to store the output results. Default: 'results'")
    parser.add_argument('--resultfile', type=str, default='hybridvit', help="Filename for storing results (without extension). Default: 'alignedcnn'")

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

    result_path = os.path.join(result_folder, result_file)

    # Define the transforms for the training set
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize([0.5], [0.5])  # Normalize the grayscale images
    ])

    matched_sets, mismatched_sets = read_pairs_file(file_path=pair_file, data_folder=data_folder)
    folds = create_folds_from_sets(matched_sets, mismatched_sets)

    bootstrap_iterations = 5  # Number of bootstrap iterations
    epochs = 1 # Number of epochs
    learning_rate = 1e-6 # Learning rate
    batch_size = 64 # Batch size
    depth_values = [2, 4, 6]  # Hyperparameter values for depth

    cross_validation_with_bootstrap(folds, transforms, bootstrap_iterations, epochs, learning_rate, batch_size, depth_values, result_path)