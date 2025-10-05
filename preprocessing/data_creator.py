import random

def read_pairs_file(file_path, data_folder='.'):
    """
    Purpose:
    ----------
    Read a file containing information about matched and mismatched image pairs and organize them into separate sets.
    This function is typically used for face verification tasks where image pairs are divided into different sets.

    Parameters:
    ----------
    - file_path: 
      The path to the text file containing information about image pairs.
      The first line of the file should specify the number of sets and the number of pairs per set.
      
    - data_folder:
      The root directory where the face images are stored. Default is the current directory ('.').

    How To:
    ----------
    1. Open the file and read all lines.
    2. Parse the number of sets and the number of pairs per set from the first line.
    3. Initialize `matched_sets` and `mismatched_sets` as empty lists to store matched and mismatched image pairs.
    4. For each set:
       a. Read `num_pairs_per_set` lines for matched pairs and add them to the `matched_set` list.
       b. Read `num_pairs_per_set` lines for mismatched pairs and add them to the `mismatched_set` list.
       c. Append the `matched_set` and `mismatched_set` to `matched_sets` and `mismatched_sets`, respectively.
    5. Repeat until all sets are read and stored.

    Output:
    ----------
    - Returns a tuple of two lists:
        - `matched_sets`: A list of lists, where each sub-list contains tuples of matched image file paths.
        - `mismatched_sets`: A list of lists, where each sub-list contains tuples of mismatched image file paths.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    num_sets, num_pairs_per_set = map(int, lines[0].strip().split())
    matched_sets = []
    mismatched_sets = []
    line_index = 1
    for _ in range(num_sets):
        matched_set = []
        mismatched_set = []
        for _ in range(num_pairs_per_set):
            name, n1, n2 = lines[line_index].strip().split()
            matched_set.append((f"{data_folder}/{name}/{name}_{int(n1):04d}.jpg", f"{data_folder}/{name}/{name}_{int(n2):04d}.jpg"))
            line_index += 1
        matched_sets.append(matched_set)

        for _ in range(num_pairs_per_set):
            name1, n1, name2, n2 = lines[line_index].strip().split()
            mismatched_set.append((f"{data_folder}/{name1}/{name1}_{int(n1):04d}.jpg", f"{data_folder}/{name2}/{name2}_{int(n2):04d}.jpg"))
            line_index += 1        
        mismatched_sets.append(mismatched_set)
    
    return matched_sets,mismatched_sets

def create_folds_from_sets(matched_sets, mismatched_sets):
    """
    Purpose:
    ----------
    The function divides each matched and mismatched set into two equal halves, creating two separate folds for each set. 
    This is used to generate training and testing folds for cross-validation in face verification tasks.

    Parameters:
    ----------
    - matched_sets: 
      A list of lists, where each sub-list contains tuples of matched face image pairs for a specific set.

    - mismatched_sets:
      A list of lists, where each sub-list contains tuples of mismatched face image pairs for a specific set.

    How To:
    ----------
    1. Create an empty list named `folds` to store the generated folds.
    2. Loop through each set in `matched_sets` and `mismatched_sets`.
    3. For each set, compute `half_size` as the length of the current matched set divided by 2.
    4. Create the first fold using the first half of `matched_sets[i]` and `mismatched_sets[i]` and add it to `folds`.
    5. Create the second fold using the remaining half of `matched_sets[i]` and `mismatched_sets[i]` and add it to `folds`.
    6. Repeat this process for each set in the input.
    
    Output:
    ----------
    - Returns a list of dictionaries where each dictionary represents a fold containing:
        - 'matched': A list of tuples representing matched face image pairs.
        - 'mismatched': A list of tuples representing mismatched face image pairs.
    """
    folds = []
    for i in range(len(matched_sets)):
        half_size = len(matched_sets[i]) // 2
        folds.append({
            'matched': matched_sets[i][:half_size],
            'mismatched': mismatched_sets[i][:half_size]
        })
        folds.append({
            'matched': matched_sets[i][half_size:],
            'mismatched': mismatched_sets[i][half_size:]
        })
    return folds

def bootstrap_632_sampling(matched_files, mismatched_files, random_state=None):

    """
    Purpose:
    ----------
    This function performs 0.632 bootstrap sampling on the given matched and mismatched file lists.
    The 0.632 bootstrap sampling is a variation of bootstrapping where 63.2% of the data is sampled with replacement 
    to create a training set, and the remaining 36.8% forms the validation set. 

    Parameters:
    ----------
    - matched_files: A list of matched face image file paths (or identifiers) to be sampled for training and validation.
      
    - mismatched_files: A list of mismatched face image file paths (or identifiers) to be sampled for training and validation.
      
    - random_state: 
      An optional integer seed for the random number generator to ensure reproducibility of the results.

    How To:
    ----------
    1. Determine the number of matched and mismatched samples (`n_matched` and `n_mismatched`).
    2. Generate training indices for matched and mismatched sets by randomly selecting 63.2% of the indices 
       with replacement using `random.randint`.
    3. Determine the validation indices as those not present in the training indices.
    4. Create the training and validation datasets using the corresponding indices for `matched_files` and `mismatched_files`.
    
    Output:
    ----------
    - Returns two dictionaries: `train_files` and `val_files`.
        - `train_files`: A dictionary containing:
            - 'matched': List of file paths/identifiers for matched pairs used in training.
            - 'mismatched': List of file paths/identifiers for mismatched pairs used in training.
          
        - `val_files`: A dictionary containing:
            - 'matched': List of file paths/identifiers for matched pairs used in validation.
            - 'mismatched': List of file paths/identifiers for mismatched pairs used in validation.
    """

    if random_state:
        random.seed(random_state)
    n_matched = len(matched_files)
    n_mismatched = len(mismatched_files)

    matched_train_indices = [random.randint(0, n_matched - 1) for _ in range(int(0.632 * n_matched))]
    mismatched_train_indices = [random.randint(0, n_mismatched - 1) for _ in range(int(0.632 * n_mismatched))]

    matched_val_indices = [i for i in range(n_matched) if i not in matched_train_indices]
    mismatched_val_indices = [i for i in range(n_mismatched) if i not in mismatched_train_indices]

    train_files = {
        'matched': [matched_files[i] for i in matched_train_indices],
        'mismatched': [mismatched_files[i] for i in mismatched_train_indices]
    }

    val_files = {
        'matched': [matched_files[i] for i in matched_val_indices],
        'mismatched': [mismatched_files[i] for i in mismatched_val_indices]
    }

    return train_files, val_files