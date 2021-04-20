from constrained_gb.metrics.binary_rates import *
from constrained_gb._constraints import *

# False negative rate <= threshold
FNR = FalseNegativeRate(block_bound=0.2)

y_true = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
# all correct
y_pred_1 = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])

print("Results for y_pred_1, when all correct,\n"
      "For this it should return -0.2 for call")

print("False Negative Rate: {}".format(false_negative(y_true, y_pred_1)))
print("Gradient of the Constraint: {}".format(FNR(y_true, y_pred_1)))
print("Penalty: {}".format(FNR.first_penalty(y_true, y_pred_1)))