from constrained_gb._constraints import FalseNegativeRate
from constrained_gb.main import ConstrainedClassifier
from constrained_gb.metrics.constrained_metrics import ConstrainedRates
import pandas as pd
from sklearn.metrics import *
from constrained_gb.metrics.binary_rates import *

def read_the_data(name):
    dir_ = "C:/Users/bam3lo/Desktop/Thesis/Dataset/"
    X_train = np.asarray(pd.read_csv(dir_ + name + "/X_train.csv")).astype('float32')
    X_test = np.asarray(pd.read_csv(dir_ + name + "/X_test.csv")).astype('float32')
    y_train = np.asarray(pd.read_csv(dir_ + name + "/y_train.csv")).astype('float32')
    y_test = np.asarray(pd.read_csv(dir_ + name + "/y_test.csv")).astype('float32')
    return X_train, y_train.ravel(), X_test, y_test.ravel()

#Heart Disease
#Breast Cancer
#Habermans Survival
#Credit card
#Diabetic
#Breast Cancer Wisconsin

dataset_name = "Diabetic"

X_train, y_train, X_test, y_test = read_the_data(dataset_name)

# thresholds = [0.1, 0.2, 0.5, 0.8]
#
# all_measures = []
#
# for t in thresholds:
#     constraints = [FalseNegativeRate(t)]
#     constrained_rate = ConstrainedRates(alpha=t, epsilon=0.05)
#
#     parms = {'constraints': constraints,
#              'multiplier_stepsize': 0.01,
#              'learning_rate': 0.1,
#              'min_samples_split': 99,
#              'min_samples_leaf': 19,
#              'max_depth': 8,
#              'max_leaf_nodes': None,
#              'min_weight_fraction_leaf': 0.0,
#              'n_estimators': 300,
#              'max_features': 'sqrt',
#              'subsample': 0.7,
#              'random_state': None
#              }
#
#     model = ConstrainedGBM(**parms)
#     model.optimize(X_train, y_train, performance_measurement=constrained_rate.Q_mean_for_fnr)
#
#     f_measures = []
#     fnrs = []
#     fprs = []
#
#     for i in range(20):
#         model.fit(X_train, y_train)
#         test_predictions = model.predict(X_test)
#         f_measures.append(f1_score(y_test, test_predictions))
#         fnrs.append(false_negative(y_test, test_predictions))
#         fprs.append(false_positive(y_test, test_predictions))
#     all_measures.append(f_measures)
#     all_measures.append(fnrs)
#     all_measures.append(fprs)
#
# df = pd.DataFrame(np.array(all_measures).T)
# df.to_csv(r"C:/Users/bam3lo/Desktop/Thesis/Dataset/" + dataset_name + ".csv")


constraints = [FalseNegativeRate(0.1)]
constrained_rate = ConstrainedRates(alpha=0.1)


parms = {'constraints': constraints,
         'multiplier_stepsize': 0.0156,
         'learning_rate': 0.164,
         'min_samples_split': 41,
         'min_samples_leaf': 11,
         'max_depth': 4,
         'max_leaf_nodes': None,
         'min_weight_fraction_leaf': 0.0,
         'n_estimators': 846,
         'max_features': 'sqrt',
         'subsample': 0.7,
         'random_state': 2
         }

# By default, we optimize 'default_hyper_parameters' list, that contains the most important
# hyper-parameters of gradient boosting by practice.
default_hyper_parameters = ['multiplier_stepsize', 'learning_rate', 'min_samples_split',
                            'min_samples_leaf', 'max_depth', 'n_estimators']

# By default, we set the domains of above hyper-parameters with default_hyper_parameters_domain list
default_hyper_parameters_domain = [(0.001, 1.0), (0.01, 1.0), (2, 100), (1, 20), (2, 9), (50, 1500)]

model = ConstrainedClassifier(**parms)
model.optimize(X_train, y_train, default_hyper_parameters, default_hyper_parameters_domain,
               performance_measurement=constrained_rate.accuracy_for_fnr, maximize=True, max_iter=3, verbosity=True)

model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
print("F Measure: {}".format(f1_score(y_test, test_predictions)))
print("FNR: {}".format(false_negative(y_test, test_predictions)))
print("FPR: {}".format(false_positive(y_test, test_predictions)))
