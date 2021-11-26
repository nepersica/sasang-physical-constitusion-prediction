import torch
import numpy as np

def numeric_score(output, label):
    """Computes scores:
    FP = False Positives -> 출혈로 오탐
    FN = False Negatives -> 실제 병변인데 prediction은 병변X
    TP = True Positives -> 실제 병변을 병변으로 예측
    TN = True Negatives -> 실제 병변x를 병변x로 예측
    return: FP, FN, TP, TN"""

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
    :param output:
    :param label:
    :return:
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


def accuracy(predicts, targets, image_size, k=1):
    batch_size = targets.size(0)
    _, ind = predicts.topk(k, 1, True, True)
    ind = torch.squeeze(ind, dim=1)
    correct = ind.eq(targets)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size / image_size / image_size)
