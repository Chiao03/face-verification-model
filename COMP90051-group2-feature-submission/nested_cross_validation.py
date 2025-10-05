import os
from preprocessing.data_creator import bootstrap_632_sampling
import csv


def folds2filenames(folds):
    """
    Purpose:
    Convert a list of fold into a single dictionary containing lists of 'matched' and 'mismatched' filenames.

    Parameters:
    - folds (list of dict): A list of dictionaries where each dictionary represents a fold and contains two keys:
                            'matched' and 'mismatched', each associated with a list of filenames.

    Result:
    Returns a dictionary with two keys:
    - 'matched' -> List of all matched filenames across all folds.
    - 'mismatched' -> List of all mismatched filenames across all folds.
    """
    filenames = {'matched': [], 'mismatched': []}
    for fold in folds:
        filenames['matched'].extend(fold['matched'])
        filenames['mismatched'].extend(fold['mismatched'])
    return filenames

def nested_cross_validation(folds, train, evaluate, param_grid, metric, fnc2vector, resultfile, bootstrap_iterations=5):
    """
    Purpose:
    Perform nested cross-validation using an outer 20-fold cross-validation loop and an inner bootstrap resampling loop
    to determine the best hyperparameters for a model and evaluate its performance.

    Parameters:
    - folds (list of dict): A list containing all folds, where each fold is a dictionary with keys 'matched' and 'mismatched'.
    - train (function): The training function that takes training samples, feature extraction function, and parameters as input.
    - evaluate (function): The evaluation function that returns the desired metric along with true labels, predictions, and scores.
    - param_grid (list of dict): List of hyperparameter dictionaries to be tested in the inner loop.
    - metric (str): The evaluation metric to use for selecting the best model in the inner loop.
    - fnc2vector (function): A function that converts raw data into feature vectors.
    - resultfile (str): The path and filename for storing results of each outer fold and the final average performance.
    - bootstrap_iterations (int, optional): Number of bootstrap iterations to perform in the inner loop. Default is 5.

    Result:
    Returns the average accuracy over all 20 outer folds and writes the results to the specified result file.
    Outputs two CSV files:
    - resultfile.csv: Contains outer fold results with best parameters and accuracies.
    - resultfile_labels.csv: Stores true and predicted labels along with prediction scores for each fold.
    """
    outer_results = []

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(resultfile), exist_ok=True)

    # Open the file to store results based on input resultfile
    with open(f"{resultfile}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write headers for outer results
        writer.writerow(["Outer Fold", "Best Params", "Inner Accuracy", "Outer Accuracy"])

        # Create a separate file for true and predicted labels, along with prediction scores
        with open(f"{resultfile}_labels.csv", mode='w', newline='') as label_file:
            label_writer = csv.writer(label_file)
            label_writer.writerow(["Outer Fold", "True Labels", "Predicted Labels", "Predicted Scores"])

            # Outer Loop: 20-fold cross-validation
            for fold_idx, test_fold in enumerate(folds):
                print(f"\n--- Outer Fold {fold_idx + 1} ---")

                # Training set from the k-1 folds
                train_folds = [folds[j] for j in range(len(folds)) if j != fold_idx]
                train_files = folds2filenames(train_folds)

                # Inner loop for hyper-parameters
                best_params = None
                best_metric = -1

                for params in param_grid:
                    avg_inner_metric = 0

                    for iteration in range(bootstrap_iterations):
                        # Bootstrap
                        train_sample, validation_sample = bootstrap_632_sampling(
                            train_files['matched'], train_files['mismatched'], random_state=iteration
                        )
                        # Bootstrap training
                        model = train(train_sample, fnc2vector, params)

                        # Bootstrap evaluation
                        inner_metric, _, _, _ = evaluate(model, validation_sample, fnc2vector, metric)
                        avg_inner_metric += inner_metric

                    avg_inner_metric /= bootstrap_iterations
                    print(f"Average Inner Accuracy for params {params}: {avg_inner_metric:.4f}")

                    # Check the best params
                    if avg_inner_metric > best_metric:
                        best_metric = avg_inner_metric
                        best_params = params

                print(f"Best params for Fold {fold_idx + 1}: {best_params} with accuracy {best_metric:.4f}")

                # Training with the best params found
                best_model = train(train_files, fnc2vector, best_params)

                # Evaluate best model on the test fold
                outer_metric, y_test, y_pred, y_score = evaluate(best_model, test_fold, fnc2vector, metric)
                outer_results.append(outer_metric)

                print(f"Outer Fold {fold_idx + 1} Accuracy: {outer_metric:.4f}")

                # Write outer results to the main results CSV
                writer.writerow([fold_idx + 1, str(best_params), f"{best_metric:.4f}", f"{outer_metric:.4f}"])

                # Change y_test to list if CNN is used
                if 'cnn' in resultfile:
                    y_test = y_test.tolist()
                    
                # Write true, predicted labels, and prediction scores to label CSV
                label_writer.writerow([
                    fold_idx + 1,
                    ','.join(map(str, y_test)),         # True Labels
                    ','.join(map(str, y_pred)),         # Predicted Labels
                    ','.join(map(str, y_score))         # Predicted Scores
                ])


            avg_outer_metric = sum(outer_results) / len(outer_results)
            print(f"\nAverage accuracy over 20 outer folds: {avg_outer_metric:.4f}")

            # Write the final average result
            writer.writerow(["Average", "-", "-", f"{avg_outer_metric:.4f}"])

    return avg_outer_metric
