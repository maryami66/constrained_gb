"""Contains functions for computing rate metrics for binary classification case.

"""

import numpy as np


def error_rate(y_true, y_pred):
    # y_true != y_pred
    signed_y_true = (y_true * 2) - 1  # -1 for negative, 1 for positive
    singed_preds = (y_pred * 2) - 1
    return np.mean(signed_y_true * singed_preds < 0.0)


def true_positive(y_true, y_pred):
    # y_true = 1, and y_pred = 1
    if np.sum(y_true > 0) == 0:  # Any positives?
        return 0.0
    else:
        return np.mean(y_pred[y_true > 0] > 0)


def false_negative(y_true, y_pred):
    # y_true = 1, but y_pred = 0
    if np.sum(y_true > 0) == 0:  # Any positives?
        return 0.0
    else:
        return np.mean(y_pred[y_true > 0] <= 0)


def true_negative(y_true, y_pred):
    # y_true = 0, and y_pred = 0
    if np.sum(y_true <= 0) == 0:  # Any negatives?
        return 0.0
    else:
        return np.mean(y_pred[y_true <= 0] <= 0)


def false_positive(y_true, y_pred):
    # y_true = 0, but y_pred = 1
    if np.sum(y_true <= 0) == 0:  # Any negatives?
        return 0.0
    else:
        return np.mean(y_pred[y_true <= 0] > 0)


def positive_rates(y_pred):
    #  = true positive + false positive
    return np.mean(y_pred > 0)


def negative_rates(y_pred):
    # = true negative + false negative
    return np.mean(y_pred <= 0)
