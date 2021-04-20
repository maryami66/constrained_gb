from RSW_HPO import SantasLittleHelper, HyperParametersOptimizer
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from constrained_gb._constraints import FalseNegativeRate
from constrained_gb.main import ConstrainedClassifier
from constrained_gb.metrics.constrained_metrics import ConstrainedRates
from sklearn.metrics import *
from constrained_gb.metrics.binary_rates import *



def read_the_data(name):
    dir_ = "C:/Users/bam3lo/Desktop/Thesis/Dataset/"
    X_train = np.asarray(pd.read_csv(dir_ + name + "/X_train.csv")).astype('float32')
    X_test = np.asarray(pd.read_csv(dir_ + name + "/X_test.csv")).astype('float32')
    y_train = np.asarray(pd.read_csv(dir_ + name + "/y_train.csv")).astype('float32')
    y_test = np.asarray(pd.read_csv(dir_ + name + "/y_test.csv")).astype('float32')
    return X_train, y_train.ravel(), X_test, y_test.ravel()

##Heart Disease
##Breast Cancer
#Habermans Survival
#Credit card
##Diabetic
#Breast Cancer Wisconsin

X_train, y_train, X_test, y_test = read_the_data("Diabetic")

constraints = [FalseNegativeRate(0.1)]
constrained_rate = ConstrainedRates(alpha=0.1)

def hpo_model_fit(
        constraints=constraints,
        multiplier_stepsize= 0.1,
        learning_rate=0.1,
        min_samples_split=358,
        min_samples_leaf=59,
        max_depth=8,
        max_leaf_nodes=None,
        min_weight_fraction_leaf=0.0,
        n_estimators=1000,
        max_features='sqrt',
        subsample=0.7,
        random_state=2,
        **kwargs
):
    score = 0
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    for train_index, test_index in folds.split(X_train_copy, y_train_copy):
        x, test_x = X_train_copy[train_index], X_train_copy[test_index]
        y, test_y = y_train_copy[train_index], y_train_copy[test_index]

        model = ConstrainedClassifier(
            constraints=constraints,
            multiplier_stepsize=multiplier_stepsize,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            n_estimators=n_estimators,
            max_features=max_features,
            subsample=subsample,
            random_state=random_state
        )

        model.fit(
            X=x,
            y=y
        )
        score += constrained_rate.accuracy_for_fnr(test_y, model.predict(test_x)) #BinaryRates().Q_mean(test_y, model.predict(test_x)) #

    perf_val = score / 5

    return perf_val


def copy_perf_summary(**kwargs):
    return True

if __name__ == '__main__':
    #################################
    opt_params = [
        #{'name': 'john', 'type': 'discrete', 'domain': ['d', 'o', 'e']},
        {'name': 'multiplier_stepsize', 'type': 'continuous', 'domain': (0.001, 1.0)},
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.01, 1.0)},
        {'name': 'min_samples_split', 'type': 'discrete', 'domain': range(2, 100)},
        {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': range(1, 20)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': range(2, 9)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': range(50, 1500)},
        #"{'name': 'block_bound', 'type': 'continuous', 'domain': (0.00, 0.15)}
    ]

    slh = SantasLittleHelper(opt_params=opt_params)
    constructor_inputs = slh.prepare_constructor_inputs()
    encoding = slh.encode_non_numerics()

    my_opt_params = HyperParametersOptimizer(
        model_fit=hpo_model_fit,
        performance_report=copy_perf_summary,
        failed_run_performance=100,
        domain_encoding=encoding,
        **constructor_inputs)

    optimal_params = my_opt_params.optimize(opt_params=opt_params, max_iter=100, metric=constrained_rate.accuracy_for_fnr)


non_opt_parms = {'constraints': constraints,
                 'max_leaf_nodes': None,
                 'min_weight_fraction_leaf': 0.0,
                 'max_features': 'sqrt',
                 'subsample': 0.7,
                 'random_state': 2}

parms = {**non_opt_parms, **optimal_params}
constrained_model = ConstrainedClassifier(**parms)

f_measures = []
fnrs = []
fprs = []

constrained_model.fit(X_train, y_train)
train_predictions = constrained_model.predict(X_train)
test_predictions = constrained_model.predict(X_test)
f_measures.append(f1_score(y_test, test_predictions))
fnrs.append(false_negative(y_test, test_predictions))
fprs.append(false_positive(y_test, test_predictions))


print("Train F1 Measure: {}".format(f1_score(y_train, train_predictions)))
print("Test F1 Measure: {} \n".format(f1_score(y_test, test_predictions)))

print("Train false negative rate: {}".format(false_negative(y_train, train_predictions)))
print("Test false negative rate: {} \n".format(false_negative(y_test, test_predictions)))

print("Train false positive rate: {}".format(false_positive(y_train, train_predictions)))
print("Test false positive rate: {} \n".format(false_positive(y_test, test_predictions)))