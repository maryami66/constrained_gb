"""Contains proxy constraints for non-decomposable constraints

When the original constraint is defined with 'g(y_true, y_pred) - a <= 0', 
wherein g(y_true, y_pred) is a rate metric or a function of several rate metrics,
then the proxy constraint is
"""

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from scipy.special import expit
from sklearn.ensemble._gb_losses import BinomialDeviance
from sklearn import metrics
from metrics.binary_rates import *

loss = BinomialDeviance(n_classes=2)


class Constraints(metaclass=ABCMeta):
    """Abstract base class for various constraints.
    
    Parameters
    ----------
    block_bound : float
        upper bound or lower bound for the constraints.
    """

    def __init__(self, block_bound):
        self.block_bound = block_bound

    @abstractmethod
    def __call__(self, y, raw_predictions):
        """Compute the measure.
        Parameters
        ----------
        y : nd-array of shape (n_samples,)
            True labels.
        raw_predictions : nd-array of shape (n_samples,)
            The prediction of the model.
        """

    @abstractmethod
    def first_penalty(self, y, raw_predictions):
        """Compute the proxy constraint penalty for first order.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : nd-array of shape (n_samples,)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """

    def second_penalty(self, tree, X, y):
        # Compute residual predictions to feed into the penalty function
        residual_predictions = tree.predict(X)
        return self.first_penalty(y, residual_predictions)


class FalseNegativeRate(Constraints, metaclass=ABCMeta):
    def __init__(self, block_bound):
        super().__init__(block_bound)
        self.block_bound = block_bound

    def __call__(self, y, raw_predictions):
        # Returns false negative rate for given labels and predictions.
        pred_proba = loss._raw_prediction_to_proba(raw_predictions)
        decision = np.argmax(pred_proba, axis=1)
        FNR = false_negative(y, decision)
        return FNR - self.block_bound

    def first_penalty(self, y, predictions):
        pred_proba = loss._raw_prediction_to_proba(predictions)
        false_negatives = np.logical_and(np.argmax(pred_proba, axis=1) != y,
                                         y == 1)
        false_positives = np.logical_and(np.argmax(pred_proba, axis=1) != y,
                                         y == 0)
        denominator = np.mean(~false_negatives) + np.mean(false_positives)

        penalty = expit(predictions.ravel()) / denominator
        return penalty

    def second_penalty(self, tree, X, y):
        # Compute residual predictions to feed into the penalty function
        residual_predictions = tree.predict(X)
        return self.first_penalty(y, residual_predictions)


class FalsePositiveRate(Constraints, metaclass=ABCMeta):
    def __init__(self, block_bound):
        super().__init__(block_bound)
        self.block_bound = block_bound

    def __call__(self, y, raw_predictions):
        # Returns false negative rate for given labels and predictions.
        pred_proba = loss._raw_prediction_to_proba(raw_predictions)
        decision = np.argmax(pred_proba, axis=1)
        FPR = false_positive(y, decision)
        return self.block_bound - FPR

    def first_penalty(self, y, predictions):
        pred_proba = loss._raw_prediction_to_proba(predictions)
        false_negatives = np.logical_and(np.argmax(pred_proba, axis=1) != y,
                                         y == 1)
        false_positives = np.logical_and(np.argmax(pred_proba, axis=1) != y,
                                         y == 0)
        denominator = np.mean(false_negatives) + np.mean(~false_positives)

        penalty = - expit(predictions.ravel()) / denominator
        return penalty

    def second_penalty(self, tree, X, y):
        # Compute residual predictions to feed into the penalty function
        residual_predictions = tree.predict(X)
        return self.first_penalty(y, residual_predictions)


class F1Measure(Constraints, metaclass=ABCMeta):
    def __init__(self, block_bound):
        super().__init__(block_bound)
        self.block_bound = block_bound

    def __call__(self, y, raw_predictions):
        #  = 2 * (precision * recall) / (precision + recall)
        pred_proba = loss._raw_prediction_to_proba(raw_predictions)
        decision = np.argmax(pred_proba, axis=1)
        return self.block_bound - metrics.f1_score(y, decision)

    def first_penalty(self, y, predictions):
        pred_proba = loss._raw_prediction_to_proba(predictions)
        false_positives = np.logical_and(np.argmax(pred_proba, axis=1) != y,
                                         y == 0)
        true_positives = np.logical_and(np.argmax(pred_proba, axis=1) == y,
                                        y == 0)
        denominator = np.mean(y == 1) + np.mean(true_positives) + np.mean(false_positives)
        penalty = expit(predictions.ravel()) / denominator
        return penalty

    def second_penalty(self, tree, X, y):
        # Compute residual predictions to feed into the penalty function
        residual_predictions = tree.predict(X)
        return self.first_penalty(y, residual_predictions)


class ExponentialLoss(Constraints, metaclass=ABCMeta):
    def __init__(self, block_bound):
        super().__init__(block_bound)
        self.block_bound = block_bound

    def __call__(self, y, raw_predictions):
        return np.mean(np.exp(-(2. * y - 1.) * raw_predictions)) - self.block_bound

    def first_penalty(self, y, predictions):
        y_ = -(2. * y - 1.)
        return y_ * np.exp(y_ * predictions.ravel())

    def second_penalty(self, tree, X, y):
        # Compute residual predictions to feed into the penalty function
        residual_predictions = tree.predict(X)
        return self.first_penalty(y, residual_predictions)

class ErrorRate(Constraints, metaclass=ABCMeta):
    def __init__(self, block_bound):
        super().__init__(block_bound)
        self.block_bound = block_bound

    def __call__(self, y, raw_predictions):
        #  = fp+fn / (fp+tp+fn+tn)
        pred_probs = loss._raw_prediction_to_proba(raw_predictions)
        decision = (np.argmax(pred_probs, axis=1) * 2) - 1
        signed_labels = (y * 2) - 1
        return np.mean(signed_labels * decision <= 0.0) - self.block_bound

    def first_penalty(self, y, predictions):
        penalty = np.zeros(predictions.shape[0])
        pred_probs = loss._raw_prediction_to_proba(predictions)
        penalty[np.argmax(pred_probs, axis=1) != y] = np.mean(np.argmax(pred_probs, axis=1) != y)
        return penalty

    def second_penalty(self, tree, X, y):
        # Compute residual predictions to feed into the penalty function
        residual_predictions = tree.predict(X)
        return self.first_penalty(y, residual_predictions)
