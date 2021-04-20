from constrained_gb.metrics.binary_rates import *
from constrained_gb.metrics.constrained_metrics import ConstrainedRates

metric = ConstrainedRates(alpha=0.2, beta=0.5, epsilon=0.05)

y_true = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
# all correct
y_pred_1 = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
# all positives
y_pred_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# all negatives
y_pred_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# two incorrects
y_pred_4 = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 0])

# four of positives are incorrects
y_pred_5 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

print("Results for y_pred_1, when all correct")

print("False Negative Rate: {}".format(false_negative(y_true, y_pred_1)))
print("False Positive Rate: {}".format(false_positive(y_true, y_pred_1)))
print("Q_mean: {}".format(metric.Q_mean_for_fnr(y_true, y_pred_1)))

print("Results for y_pred_2, when all positives")

print("False Negative Rate: {}".format(false_negative(y_true, y_pred_2)))
print("False Positive Rate: {}".format(false_positive(y_true, y_pred_2)))
print("Q_mean: {}".format(metric.Q_mean_for_fnr(y_true, y_pred_2)))

print("Results for y_pred_3, when all negatives")

print("False Negative Rate: {}".format(false_negative(y_true, y_pred_3)))
print("False Positive Rate: {}".format(false_positive(y_true, y_pred_3)))
print("Q_mean: {}".format(metric.Q_mean_for_fnr(y_true, y_pred_3)))

print("Results for y_pred_4, only two incorrects")

print("False Negative Rate: {}".format(false_negative(y_true, y_pred_4)))
print("False Positive Rate: {}".format(false_positive(y_true, y_pred_4)))
print("Q_mean: {}".format(metric.Q_mean_for_fnr(y_true, y_pred_4)))

print("Results for y_pred_5, four of positives are incorrects")

print("False Negative Rate: {}".format(false_negative(y_true, y_pred_5)))
print("False Positive Rate: {}".format(false_positive(y_true, y_pred_5)))
print("Q_mean: {}".format(metric.Q_mean_for_fnr(y_true, y_pred_5)))