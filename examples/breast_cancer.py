import gradient_boosting_constrained_optimization as gbmco
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)


constraints = [gbmco.FalseNegativeRate(0.001)]

parms = {'constraints': constraints,
         'multiplier_stepsize': 0.01,
         'learning_rate': 0.1,
         'min_samples_split': 99,
         'min_samples_leaf': 19,
         'max_depth': 8,
         'max_leaf_nodes': None,
         'min_weight_fraction_leaf': 0.0,
         'n_estimators': 300,
         'max_features': 'sqrt',
         'subsample': 0.7,
         'random_state': 2
         }

clf = gbmco.ConstrainedClassifier(**parms)
clf.fit(X_train, y_train)

test_predictions = clf.predict(X_test)

print("Test F1 Measure: {} \n".format(f1_score(y_test, test_predictions)))
print("Test FNR: {} \n".format(1-recall_score(y_test, test_predictions)))