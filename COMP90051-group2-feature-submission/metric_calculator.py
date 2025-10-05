
def accuracy(y_true, y_pred):
    """
    Computes the accuracy of predictions.

    Parameters:
    y_true (list or array): List or array of true labels.
    y_pred (list or array): List or array of predicted labels.

    Returns:
    float: Accuracy of the predictions as a float between 0 and 1.

    Raises:
    ValueError: If the lengths of y_true and y_pred are not the same.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The length of true labels and predicted labels must be the same.")

    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    acc = correct / len(y_true)
    return acc