"""Contains 'ConstrainedGBM' class.

This method is a subclass of SKlearn gradient boosting classifier

The Lagrangian serves as a loss function for gradient boosting classifier.

"""

from abc import ABCMeta
from abc import abstractmethod
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._gb import GradientBoostingClassifier
from sklearn.ensemble._gb import GradientBoostingRegressor
from sklearn.dummy import DummyClassifier
from constrained_gb.lagrangian import ProxyLagrangianBinomialDeviance
from constrained_gb.lagrangian import ProxyLagrangianLeastSquaresError
from constrained_gb.hyperparameter_optimization.bayesian_optimization import HyperParameterOptimization


class BaseConstrainedGBM(BaseGradientBoosting, metaclass=ABCMeta):
    """Abstract base class for Constrained Gradient Boosting which is
    a subclass of SKlearn BaseGradientBoosting class.
    """

    @abstractmethod
    def __init__(self, constraints, multiplier_stepsize, update_type,
                 multipliers_radius, learning_rate, n_estimators, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth,
                 min_impurity_decrease, min_impurity_split, init, subsample, max_features,
                 ccp_alpha, random_state, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='deprecated', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):

        self.constraints = constraints
        self.multiplier_stepsize = multiplier_stepsize
        self.update_type = update_type
        self.multipliers_radius = multipliers_radius

        self.lagrangian = ProxyLagrangianBinomialDeviance(self.constraints,
                                                          self.multiplier_stepsize,
                                                          self.update_type,
                                                          self.multipliers_radius)
        super().__init__(
            loss='constrained_deviance', learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, presort=presort,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

    def _check_params(self):
        try:
            super()._check_params()
        except ValueError as e:
            if str(e) == "Loss 'constrained_deviance' not supported. ":
                self.loss_ = self.lagrangian
            elif str(e) == "Loss 'constrained_lse' not supported. ":
                self.loss_ = self.lagrangian
            else:
                raise
        if self.multiplier_stepsize <= 0.0:
            raise ValueError("Multiplier stepsize must be greater than 0 but "
                             "was %r" % self.multiplier_stepsize)

    def optimize(self, X, y, hyper_parameters, hyper_parameters_domain,
                 performance_measurement, nb_folds=5, max_iter=100,
                 maximize=False, random_state=None, verbosity=False):
        """Hyper-parameter optimization of gradient boosting classifier,
            using bayesian optimization.
           It sets the hyper-parameters of the gradient boosting classifier
           to optimum hyper-parameters.
        Parameters
        ----------
        X: nd-array of shape (n_samples, n_features)
            The input samples of the training set.
        y: nd-array of shape (n_samples,)
            The target values of the training set.
        hyper_parameters: list of strings, default = None
            The list of hyper-parameter that we desire to optimize. If None,
            it optimizes the default_hyper_parameters list.
        hyper_parameters_domain: list of tuples or lists, default = None
            The list of hyper parameters domains that be correspondent with
            hyper_parameters list. For integer and float hyper-parameters
            we set a tuple or lower and upper bounds of the corresponding
            hyper-parameter. But, for the string hyper-parameters such as
            'max_features', we set a list of possible ones.
            If None, it optimizes the default_hyper_parameters_domain list.
        performance_measurement: function, default = None
            The evaluation metric that we desire to optimize the hyper-parameters
            based on. If None, it maximizes f1_score.
        nb_folds: integer, default = 5
            The number of folds for k-fold cross validation.
        max_iter: integer, default = 100
            The maximum number of iterations for Bayesian Optimization
        maximize: bool, default = False
            If True, we minimize the -performance measurement.
        random_state: integer, default = None
            The random seed controller for the SKlearn KFold.
        verbosity: bool, default = False
            If True, print out the tryout values and the performance measurement.
        """
        hp_opt = HyperParameterOptimization(gradient_boosting_model=self,
                                            hyper_parameters=hyper_parameters,
                                            hyper_parameters_domain=hyper_parameters_domain,
                                            performance_measurement=performance_measurement)
        optimized_parameters = hp_opt.optimize(X, y, max_iter=max_iter, maximize=maximize,
                                               nb_folds=nb_folds, random_state=random_state,
                                               verbosity=verbosity)

        self.set_params(**optimized_parameters)


class ConstrainedClassifier(BaseConstrainedGBM, GradientBoostingClassifier):
    """
    Parameters
    ----------
    constraints: list of rate 
    FNR : False Negative Rate upper bound
    multiplier_stepsize : float, default=0.01
        learning rate to update Lagrangian multiplier
    radius: float, default=1.0
        The radius of Lagrangian Multiplier space
    multiplier_init: float, default=0.0
        Initialized multiplier
    Attributes
    ----------
    """

    _SUPPORTED_LOSS = ('constrained_deviance')

    def __init__(self, constraints, multiplier_stepsize, update_type='multiplicative',
                 multipliers_radius=1., learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse',
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0., min_impurity_split=None,
                 init=DummyClassifier(strategy='prior'), random_state=None,
                 max_features=None, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='deprecated', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):
        self.constraints = constraints
        self.multiplier_stepsize = multiplier_stepsize
        self.update_type = update_type
        self.multipliers_radius = multipliers_radius

        self.lagrangian = ProxyLagrangianBinomialDeviance(self.constraints,
                                                          self.multiplier_stepsize,
                                                          self.update_type,
                                                          self.multipliers_radius)

        super().__init__(
            constraints=constraints, multiplier_stepsize=multiplier_stepsize,
            update_type=update_type, multipliers_radius=multipliers_radius,
            learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, presort=presort,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)


class ConstrainedRegressor(BaseConstrainedGBM, GradientBoostingRegressor):
    """Gradient Boosting for regression.
    """

    _SUPPORTED_LOSS = ('constrained_lse')

    def __init__(self, constraints, multiplier_stepsize, update_type='regular',
                 multipliers_radius=1., learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='deprecated',
                 validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):
        self.constraints = constraints
        self.multiplier_stepsize = multiplier_stepsize
        self.update_type = update_type
        self.multipliers_radius = multipliers_radius

        self.lagrangian = ProxyLagrangianLeastSquaresError(self.constraints,
                                                          self.multiplier_stepsize,
                                                          self.update_type,
                                                          self.multipliers_radius)

        super().__init__(
            constraints=constraints, multiplier_stepsize=multiplier_stepsize,
            update_type=update_type, multipliers_radius=multipliers_radius,
            learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            presort=presort, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)
