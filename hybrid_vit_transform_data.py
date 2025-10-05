import os
import random

def transform_data(dict):
    '''
    Description:
    Transforms the data from the dictionary format to a list of tuples containing image paths and labels

    Parameters:
    - dict (dict): Dictionary containing matched and mismatched pairs

    Returns:
    - transformed_data (list): List of tuples containing image paths and labels
    '''
    person_to_label = {}
    current_label = 0

    list_of_pairs = dict['matched'] + dict['mismatched']

    # For each pair of images, assign a unique label to each person
    for img1, img2 in list_of_pairs:
        person1 = os.path.basename(os.path.dirname(img1))
        person2 = os.path.basename(os.path.dirname(img2))
        
        if person1 not in person_to_label:
            person_to_label[person1] = current_label
            current_label += 1
        if person2 not in person_to_label:
            person_to_label[person2] = current_label
            current_label += 1

    transformed_data = []

    # Create a list of tuples containing image paths and labels
    for img1, img2 in list_of_pairs:
        person1 = os.path.basename(os.path.dirname(img1))
        person2 = os.path.basename(os.path.dirname(img2))
        label1 = person_to_label[person1]
        label2 = person_to_label[person2]
        
        transformed_data.append((img1, label1))
        transformed_data.append((img2, label2))

    # Shuffle the data
    random.shuffle(transformed_data)

    return transformed_data