from sklearn.metrics import *
from constrained_gb.metrics.binary_rates import *


class ConstrainedRates:
    def __init__(self, alpha, beta=20):
        self.alpha = alpha
        self.beta = beta

    def Q_mean_for_fnr(self, y_true, y_pred):
        fnr = false_negative(y_true, y_pred)
        fpr = false_positive(y_true, y_pred)
        if fnr <= self.alpha:
            return np.sqrt(fnr ** 2 + fpr ** 2)
        else:
            return np.sqrt(fnr ** 2 + fpr ** 2) + self.beta * (fnr - self.alpha)

    def Q_mean_for_fpr(self, y_true, y_pred):
        fnr = false_negative(y_true, y_pred)
        fpr = false_positive(y_true, y_pred)
        if fpr <= self.alpha:
            return np.sqrt(fnr ** 2 + fpr ** 2)
        else:
            return np.sqrt(fnr ** 2 + fpr ** 2) + self.beta * (fpr - self.alpha)

    def accuracy_for_fnr(self, y_true, y_pred):
        fnr = false_negative(y_true, y_pred)
        if fnr <= self.alpha:
            return 1 - accuracy_score(y_true, y_pred)
        else:
            return (1 - accuracy_score(y_true, y_pred)) + (self.beta * (fnr - self.alpha))

    def accuracy_for_fpr(self, y_true, y_pred):
        fpr = false_positive(y_true, y_pred)
        if fpr <= self.alpha:
            return accuracy_score(y_true, y_pred)
        else:
            return accuracy_score(y_true, y_pred) + self.beta * (fpr - self.alpha)

    def G_mean_for_fnr(self, y_true, y_pred):
        fnr = false_negative(y_true, y_pred)
        fpr = false_positive(y_true, y_pred)
        if fnr <= self.alpha:
            return np.sqrt(fnr * fpr)
        else:
            return np.sqrt(fnr * fpr) + self.beta * (fnr - self.alpha)

    def G_mean_for_fpr(self, y_true, y_pred):
        fnr = false_negative(y_true, y_pred)
        fpr = false_positive(y_true, y_pred)
        if fpr <= self.alpha:
            return np.sqrt(fnr * fpr)
        else:
            return np.sqrt(fnr * fpr) + self.beta * (fpr - self.alpha)

    def F1_measure(self, y_true, y_pred):
        f1_measure = f1_score(y_true, y_pred)
        if f1_measure >= self.alpha:
            return f1_measure
        else:
            return f1_measure + self.beta
