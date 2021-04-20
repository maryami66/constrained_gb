"""Contains 'HyperParameterOptimization' class.

By taking advantages of GPyOpt BayesianOptimization, we optimize the gradient boosting classifier hyper-parameters
"""


from sklearn.model_selection import StratifiedKFold
from GPyOpt.methods import BayesianOptimization
import copy
from sklearn.metrics import *


class HyperParameterOptimization:
    """A class for hyper-parameter optimization of gradient boosting classifier,
    using bayesian optimization.
    Parameters
    ----------
    gradient_boosting_model : ConstrainedGBM object
        upper bound or lower bound for the constraints.
    hyper_parameters: list of strings, default = default_hyper_parameters
        The list of hyper-parameter that we desire to optimize
    hyper_parameters_domain: list of tuples or lists, default = default_hyper_parameters_domain
        The list of hyper parameters domains that be correspondent with
        hyper_parameters list. For integer and float hyper-parameters
        we set a tuple or lower and upper bounds of the corresponding
        hyper-parameter. But, for the string hyper-parameters such as
        'max_features', we set a list of possible ones.
    performance_measurement: function, default = f1-measure
        The evaluation metric that we desire to optimize the hyper-parameters
        based on.
    """

    def __init__(self, gradient_boosting_model,
                 hyper_parameters,
                 hyper_parameters_domain,
                 performance_measurement):
        self.model = gradient_boosting_model
        self.hyper_parameters = hyper_parameters
        self.hyper_parameters_domain = hyper_parameters_domain
        self.performance_measurement = performance_measurement

        # Take all the hyper-parameters
        self.all_hyper_parameters = self.model.get_params()

        # Get the hyper-parameters that are fixed by user
        self.fixed_hyper_parameters = {k: v for k, v in self.all_hyper_parameters.items()
                                       if k not in self.hyper_parameters}

        # We only optimize the following list as the others is user-dependant
        # We use this list to raise error if the selected hyper-parameters
        # are not in.
        self.we_only_optimize = ['learning_rate', 'multiplier_stepsize',
                                 'min_samples_split', 'min_samples_leaf',
                                 'max_leaf_nodes', 'max_depth', 'max_features',
                                 'min_impurity_decrease', 'min_weight_fraction_leaf',
                                 'n_estimators', 'criterion']
        # The following hyper-parameters are integer, we use this in
        # bayesian optimization of GPyOpt.
        self.integer_params = ['min_samples_split', 'min_samples_leaf',
                               'max_leaf_nodes', 'max_depth', 'n_estimators']
        # same as above for the string hyper-parameters.
        self.string_params = ['max_features', 'criterion']

    def _check_parameters(self):
        """Check validity of parameters and raise Error if not valid.
        """
        # check if the selected hyper-parameters belong to gradient boosting
        for param in self.hyper_parameters:
            if param not in self.all_hyper_parameters.keys():
                raise NameError("%r is not a hyper-parameter for gradient boosting" % param)

        # Ceck if the size of hyper-parameters list and the domain list is equal or not
        if len(self.hyper_parameters) != len(self.hyper_parameters_domain):
            raise ValueError("The size of hyper-parameters list is not equal to "
                             "the size of hyper-parameters domain list.\n"
                             "The number of hyper-parameters: %d, but the "
                             "number of hyper-parameter domains: "
                             "%d" % (len(self.hyper_parameters),
                                     len(self.hyper_parameters_domain)))
        # Check if the selected hyper-parameters belong to the list of
        # hyper-parameters that we optimize or not
        for param in self.hyper_parameters:
            if param not in self.we_only_optimize:
                raise ValueError("We do not optimize %r,"
                                 "we only optimize %r" % (param, self.we_only_optimize))
        # Check the hyper-parameters domain list
        for i, param in enumerate(self.hyper_parameters):
            # For integer hyper-parameters we check if the domain is a tuple of
            # lower and upper bound, and if it is a tuple of integers
            if param in self.integer_params:
                if type(self.hyper_parameters_domain[i]) != tuple:
                    raise ValueError("The corresponding domain for %r should be a tuple,"
                                     "but it is %r" %(param, type(self.hyper_parameters_domain[i])))
                if all(type(d) == 'int' for d in self.hyper_parameters_domain[i]):
                    raise ValueError("The corresponding domain for %r should be a tuple"
                                     "of integers, but it is a tuple of floats" % param)
            # For string hyper-parameters we check if the domain is a list,
            # and if it is a list of valid values or not.
            elif param in self.string_params:
                if type(self.hyper_parameters_domain[i]) != list:
                    raise ValueError("The corresponding domain for %r should be a list of"
                                     "strings but it is %r,"
                                     "" % (param, type(self.hyper_parameters_domain[i])))
                if param == 'max_features':
                    if set(self.hyper_parameters_domain[i]).issubset({"auto", "sqrt", "log2"}):
                        raise ValueError("Invalid domain for max_features: %r,"
                                         " the domain should be a subset of "
                                         "['auto', 'sqrt', 'log2']"
                                         "" % self.hyper_parameters_domain[i])
                if param == 'criterion':
                    if set(self.hyper_parameters_domain[i]).issubset({"friedman_mse", "mse", "mae"}):
                        raise ValueError("Invalid domain for criterion: %r,"
                                         " the domain should be a subset of "
                                         "['friedman_mse', 'mse', 'mae']"
                                         "" % self.hyper_parameters_domain[i])
            # For float hyper-parameters we check if the domain is a tuple or not.
            else:
                if type(self.hyper_parameters_domain[i]) != tuple:
                    raise ValueError("The corresponding domain for %r should be a tuple,"
                                     "but it is %r" %(param, type(self.hyper_parameters_domain[i])))

    def _create_opt_params(self):
        """Generate a list of dictionaries containing the description of the inputs variables
        using hyper_parameters list and hyper_parameters_domain list to feed in to the domain
        argument of GPyOpt bayesian optimization

        Returns
        -------
        domains: a list of dictionaries
        """
        domains = []
        # check the validity of the parameters
        self._check_parameters()
        # for each selected hyper-parameters build a dictionary.
        for i, param in enumerate(self.hyper_parameters):
            if param in self.integer_params:
                param_dict = {'name': param, 'type': 'discrete',
                              'domain': range(self.hyper_parameters_domain[i][0],
                                              self.hyper_parameters_domain[i][1])}
            elif param in self.string_params:
                param_dict = {'name': param, 'type': 'discrete',
                              'domain': list(range(len(self.hyper_parameters_domain[i])))}
            else:
                param_dict = {'name': param, 'type': 'continuous',
                              'domain': self.hyper_parameters_domain[i]}
            domains.append(param_dict)
        return domains

    def _separate_parameters(self, tryout_params):
        """Prepare hyper-parameters dictionary to feed in to the gradient boosting model

        Parameters
        ----------
        tryout_params: 2d-array of shape (1, n).
            At each try, GPyOpt BayesianOptimization takes a set of parameters to try out
            the objective function, which is a 2-dimensional array of shape (1, n), where
            n is the number of selected hyper-parameters for optimization.

        Returns
        -------
        params: dictionary
            A dictionary of gradient boosting hyper-parameters as keys that have
            tryout values of BayesianOptimization.
        """
        optimizable_hyper_parameters = {}
        for i, param in enumerate(self.hyper_parameters):
            if param in self.integer_params:
                optimizable_hyper_parameters[param] = int(tryout_params[0, i])
            elif param in self.string_params:
                if param == 'max_features':
                    param_list = self.hyper_parameters_domain[
                        self.hyper_parameters.index('max_features')]
                elif param == 'criterion':
                    param_list = self.hyper_parameters_domain[
                        self.hyper_parameters.index('criterion')]
                optimizable_hyper_parameters[param] = param_list[int(tryout_params[0, i])]
            else:
                optimizable_hyper_parameters[param] = tryout_params[0, i]
        # Combine the fix-valued hyper-parameters and the selected hyper parameters
        # to optimize with try out values.
        params = {**self.fixed_hyper_parameters, **optimizable_hyper_parameters}
        return params

    def _decode_parameters(self, optimum_params):
        """Assign the list of optimum values to the corresponding hyper-parameters.
            It only is used for returning the optimum hyper-parameters.

        Parameters
        ----------
        optimum_params: 2d-array of shape (1, n).
            At each try, GPyOpt BayesianOptimization takes a set of parameters to try out
            the objective function, which is a 2-dimensional array of shape (1, n), where
            n is the number of selected hyper-parameters for optimization.

        Returns
        -------
        params: dictionary
            A dictionary of gradient boosting hyper-parameters as keys that have
            tryout values of BayesianOptimization.

        Notes
        -------
        The difference between this method and the _separate_parameters method
        is the input. The input of the _separate_parameters is a 2d- array but
        the input of this method is a list of floats.
        """
        optimizable_hyper_parameters = {}
        for i, param in enumerate(self.hyper_parameters):
            if param in self.integer_params:
                optimizable_hyper_parameters[param] = int(optimum_params[i])
            elif param in self.string_params:
                if param == 'max_features':
                    param_list = self.hyper_parameters_domain[
                        self.hyper_parameters.index('max_features')]
                elif param == 'criterion':
                    param_list = self.hyper_parameters_domain[
                        self.hyper_parameters.index('criterion')]
                optimizable_hyper_parameters[param] = param_list[int(optimum_params[i])]
            else:
                optimizable_hyper_parameters[param] = optimum_params[i]
        # Combine the fix-valued hyper-parameters and the selected hyper parameters
        # to optimize with try out values.
        params = {**self.fixed_hyper_parameters, **optimizable_hyper_parameters}
        return params

    def _model_fit(self, gb_params, X_train, y_train, nb_folds=5, random_state=None):
        """Apply k-fold cross-validation for the training set.

        Parameters
        ----------
        gb_params: dictionary
            The hyper-parameters of the gradient boosting model.
        X_train: nd-array of shape (n_samples, n_features)
            The input samples of the training set.
        y_train: nd-array of shape (n_samples,)
            The target values of the training set.
        nb_folds: integer, default = 5
            The number of folds for k-fold cross validation.
        random_state: integer, default = None
            The random seed controller for the SKlearn KFold.
        Returns
        -------
        performance_evaluation: float
            The average performance of the gradient boosting method that is
            evaluated by performance measurement.
        """
        score = 0
        model = self.model.set_params(**gb_params)
        # Shuffle the training set and create K folds.
        folds = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=random_state)
        for train_index, test_index in folds.split(X_train, y_train):
            x, test_x = X_train[train_index], X_train[test_index]
            y, test_y = y_train[train_index], y_train[test_index]

            # For each fold, fit a gradient boosting model.
            model.fit(
                X=x,
                y=y
            )

            # Record the performance evaluation of the model.
            score += self.performance_measurement(test_y, model.predict(test_x))

        # take the average of the performances
        performance_evaluation = score / nb_folds

        return performance_evaluation

    def optimize(self, X_train, y_train, nb_folds=5,
                 max_iter=100, maximize=False, random_state=None,
                 verbosity=False):
        """Bayesian Optimization for Gradient Boosting Model to optimize the hyper-parameters
            This is the main method that apply GPyOpt BayesianOptimization.

        Parameters
        ----------
        X_train: nd-array of shape (n_samples, n_features)
            The input samples of the training set.
        y_train: nd-array of shape (n_samples,)
            The target values of the training set.
        nb_folds: integer, default = 5
            The number of folds for k-fold cross validation.
        max_iter: integer, default = 100
            The maximum number of iterations for Bayesian Optimization
        maximize: bool, default = False
            If True, we minimize the -performance measurement.
        verbosity: bool, default = False
            If True, print out the tryout values and the performance measurement.

        Returns
        -------
        decoded_hyper_parameters: dictionary
            A dictionary of gradient boosting hyper-parameters with optimum
            values.
        """
        # For keeping track of Bayesian Optimization iterations. It uses
        # just for printing out the trails.
        self.number_of_iters = 0
        # Prepare the domains for GPyOpt BayesianOptimizations
        domains = self._create_opt_params()
        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()

        def _model_fit_for_bayesian_optimization(tryout_params):
            """Function to optimize.
                It should take 2-dimensional numpy arrays as input and return 2-dimensional outputs.
            """
            self.number_of_iters += 1

            params = self._separate_parameters(tryout_params)


            performance = self._model_fit(params, X_train_copy, y_train_copy,
                                          nb_folds=nb_folds, random_state=random_state)

            if verbosity:
                msg = 'Starting HPO run: {}'
                print(msg.format(self.number_of_iters))
                print('Evaluating hyper parameter combination:')

                for key, value in params.items():
                    if key in self.hyper_parameters:
                        msg = '{} : {}'.format(key, value)
                        print(msg)
                print("Performance: {}".format(performance))
                print('-' * 30)

            return performance
        
        # Run the Bayesian Optimization
        my_opt_params = BayesianOptimization(f=_model_fit_for_bayesian_optimization,
                                             domain=domains,
                                             maximize=maximize)
        
        my_opt_params.run_optimization(max_iter=max_iter)
        
        # Convert the list of optimum values to dictionary
        decoded_hyper_parameters = self._decode_parameters(my_opt_params.x_opt)
        return decoded_hyper_parameters
