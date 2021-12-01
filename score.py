import torch
import numpy as np

def numeric_score(output, label):
    """
    Computes scores
    """
    FP = np.float(np.sum((output == 1) & (label == 0)))
    FN = np.float(np.sum((output == 0) & (label == 1)))
    if FP != 0.0 or FN != 0.0:
        pass
    TP = np.float(np.sum((output == 1) & (label == 1)))
    TN = np.float(np.sum((output == 0) & (label == 0)))

    return FP, FN, TP, TN

def get_score(output, label):
    """
    get score based on confusion matrix
    """
    FP, FN, TP, TN = numeric_score(output, label)
    N = FP + FN + TP + TN

    epsilon = 1e-5

    # Recall : TP / TP+FN
    recall = np.divide(TP, TP + FN + epsilon)
    # Precision : TP / TP+FP
    precision = np.divide(TP, (TP+FP+epsilon))

    accuracy = np.divide((TP + TN), N+epsilon)

    # F1 socre = 2 * (A interect B) / |A| + |B| = 2TP / 2TP + FP + FN
    f1_score = 2 * (precision*recall) / (precision + recall + epsilon)
    dice_coeff = 2*TP / (2*TP+FP+FN+epsilon)

    # J(A,B) = | A intersect B | / | A union B |
    jaccard_score = TP / (TP+FN+FP+ epsilon)

    return recall * 100, precision * 100, accuracy * 100, f1_score*100, jaccard_score*100