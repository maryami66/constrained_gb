from constrained_gb.metrics.binary_rates import *

y_true = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
# all correct
y_pred_1 = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
# all positives
y_pred_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# all negatives
y_pred_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# two incorrects
y_pred_4 = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 0])

print("Results for y_pred_1, when all correct")

print("False Positive Rate: {}".format(false_positive(y_true, y_pred_1)))
print("False Negative Rate: {}".format(false_negative(y_true, y_pred_1)))
print("True Positive Rate: {}".format(true_positive(y_true, y_pred_1)))
print("True Negative Rate: {}".format(true_negative(y_true, y_pred_1)))
print("Positive Rate: {}".format(positive_rates(y_pred_1)))
print("Negative Rate: {}".format(negative_rates(y_pred_1)))

print("Results for y_pred_2, when all positives")

print("False Positive Rate: {}".format(false_positive(y_true, y_pred_2)))
print("False Negative Rate: {}".format(false_negative(y_true, y_pred_2)))
print("True Positive Rate: {}".format(true_positive(y_true, y_pred_2)))
print("True Negative Rate: {}".format(true_negative(y_true, y_pred_2)))
print("Positive Rate: {}".format(positive_rates(y_pred_2)))
print("Negative Rate: {}".format(negative_rates(y_pred_2)))

print("Results for y_pred_3, when all negatives")

print("False Positive Rate: {}".format(false_positive(y_true, y_pred_3)))
print("False Negative Rate: {}".format(false_negative(y_true, y_pred_3)))
print("True Positive Rate: {}".format(true_positive(y_true, y_pred_3)))
print("True Negative Rate: {}".format(true_negative(y_true, y_pred_3)))
print("Positive Rate: {}".format(positive_rates(y_pred_3)))
print("Negative Rate: {}".format(negative_rates(y_pred_3)))

print("Results for y_pred_4, only two incorrects")

print("False Positive Rate: {}".format(false_positive(y_true, y_pred_4)))
print("False Negative Rate: {}".format(false_negative(y_true, y_pred_4)))
print("True Positive Rate: {}".format(true_positive(y_true, y_pred_4)))
print("True Negative Rate: {}".format(true_negative(y_true, y_pred_4)))
print("Positive Rate: {}".format(positive_rates(y_pred_4)))
print("Negative Rate: {}".format(negative_rates(y_pred_4)))