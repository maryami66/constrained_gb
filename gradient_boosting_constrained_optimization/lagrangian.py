"""Contains 'ProxyLagrangian' class.

This class is a subclass of SKlearn Binomial Deviance for non-decomposable constraints
"""


import numpy as np
from scipy.special import expit
from sklearn.tree._tree import TREE_LEAF
from sklearn.ensemble._gb_losses import BinomialDeviance
from sklearn.ensemble._gb_losses import LeastSquaresError
import gradient_boosting_constrained_optimization._utilities as utils


class ProxyLagrangianBinomialDeviance(BinomialDeviance):
    """Proxy Lagrangian formulation for Binomial Deviance loss function.
    
    Parameters
    ----------
    constraints: constraint or list of constraints, default=None
        The constraints should be an instance of a class which
        has to provide  :meth:`__call__`, and :meth:`penalty`.
        If None, the classifier is unconstrained optimization.
        are the evaluation measurements such as F1_measures
    multiplier_stepsize : float, default=0.01,
        multiplier stepsize is the learning rate for gradient ascend.
    update_type: string, default='multiplicative',
        The update type of multiplier is either multiplicative or additive.
        The multiplicative updates is used for non-decomposable constraints.
        Since for classification case, constraints are mainly non-decomposable,
        the default update type is multiplicative.
    multipliers_radius: float, default=1.,
        If update_type='additive', then the multipliers projects to the
        l_1-norm ball with radius of 'multipliers_radius'. Otherwise, it
        has no effects.
    """

    def __init__(self,
                 constraints=None,
                 multiplier_stepsize=0.01,
                 update_type='multiplicative',
                 multipliers_radius=1.):
        super().__init__(n_classes=2)

        # when the constraint is None, the problem is unconstrained
        if constraints is None:
            pass
        # otherwise, we consider a list of all constraints
        else:
            if not isinstance(constraints, list):
                self.constraints = [constraints]
            else:
                self.constraints = constraints

        self.multiplier_stepsize = multiplier_stepsize
        self.update_type = update_type
        self.multipliers_radius = multipliers_radius

        # number of constraints
        self.nb_constraints = len(self.constraints)

        # we start with zero for Lagrange multipliers
        # but one for the objective multiplier.
        self.multipliers = np.zeros((self.nb_constraints + 1, 1))
        self.multipliers[0, 0] = 1

        # if the update type is multiplicative, we start the distribution
        # with M_(i,j) = 1/(m+1), where, m is the number of constraints
        # otherwise, it has no effects.
        self.multipliers_distribution = np.ones((self.nb_constraints + 1,
                                                 self.nb_constraints + 1)
                                                ) / self.nb_constraints + 1

    def _gradient_wrt_multiplier(self, y, raw_predictions):
        """Computes gradients of the Lagrangian with respect to multipliers. """
        # zero for objective is padded
        gradient_wrt_multiplier = np.zeros((self.nb_constraints+1, 1))

        # for each constraint we compute the constraint violation
        for i in range(self.nb_constraints):
            gradient_wrt_multiplier[i+1, 0] = self.constraints[i](y, raw_predictions)
        return gradient_wrt_multiplier

    def _first_order_penalties(self, y, raw_predictions):
        """Computes the penalty term for negative gradient. """
        # For each constraint we compute the penalty term, then multiply it
        # by Lagrange multiplier.
        proxy_penalties = np.array(
            [self.constraints[i].first_penalty(y, raw_predictions) for i in range(self.nb_constraints)])
        #  At the end, we sum up all penalties.
        sum_penalties = np.sum(np.multiply(self.multipliers[1:], proxy_penalties), axis=0)
        return sum_penalties

    def _second_order_penalties(self, tree, X, y):
        """Computes the penalty term for the tree leaves. """
        # For each constraint we compute the penalty term for residual predictions,
        # then multiply it by Lagrange multiplier.
        proxy_penalties = np.array(
            [self.constraints[i].second_penalty(tree, X, y) for i in range(self.nb_constraints)])
        #  At the end, we sum up all penalties.
        sum_penalties = np.sum(np.multiply(self.multipliers[1:], proxy_penalties), axis=0)
        return sum_penalties

    def _update_multipliers(self, y, raw_predictions):
        """Updates the Lagrange multipliers and the multipliers distribution,
        Raises:
            ValueError: if update type is neither multiplicative or additive
        """
        # Computes the gradient of the Lagrangian wrt to the Lagrange multipliers
        gradient_wrt_multiplier = self._gradient_wrt_multiplier(y, raw_predictions)

        # If update type is multiplicative, updates the Lagrange multipliers
        # and the multipliers distribution
        if self.update_type == 'multiplicative':
            self.multipliers_distribution, self.multipliers = utils._multiplier_multiplicative_update(
                self.multipliers_distribution,
                self.multipliers,
                gradient_wrt_multiplier,
                self.multiplier_stepsize)

        # If update type is additive, updates  only the Lagrange multipliers
        elif self.update_type == 'additive':
            self.multipliers = utils._multiplier_additive_update(
                    self.multipliers,
                    gradient_wrt_multiplier,
                    self.multiplier_stepsize,
                    self.multipliers_radius)
        # Otherwise, raises value error.
        else:
            raise ValueError(
                    "update_type must be 'multiplicative' or 'additive' not %d" %
                    self.update_type)

    def negative_gradient(self, y, raw_predictions, **kwargs):
        """Compute the gradient with respect to the residual (= negative gradient).
            Negative gradient is the negative gradient of proxy Lagrangian,
            which is
        Parameters
        ----------
        **kwargs
        y : 1d-array, shape (n_samples,)
            True labels.
        raw_predictions : 2d-array, shape (n_samples, K)
            The raw_predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        # computes negative gradient
        loss_gradient = self.multipliers[0, 0] * (
                y - expit(raw_predictions.ravel()))
        # Sum up the objectives and the constraints
        return loss_gradient + self._first_order_penalties(y, raw_predictions)

    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.
        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : nd-array of shape (n_samples, n_features)
            The data array.
        y : nd-array of shape (n_samples,)
            The target labels.
        residual : nd-array of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : nd-array of shape (n_samples, )
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : nd-array of shape (n_samples,)
            The weight of each sample.
        sample_mask : nd-array of shape (n_samples,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            model_stepsize shrinks the contribution of each tree by
            ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.
        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, y, residual,
                                         raw_predictions[:, k], sample_weight)

        # update predictions (both in-bag and out-of-bag)
        raw_predictions[:, k] += \
            learning_rate * tree.value[:, 0, 0].take(terminal_regions, axis=0)

        self._update_multipliers(y, raw_predictions)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):

        # residual predictions for the second order penalty
        second_order_penalty = sample_weight * self._second_order_penalties(tree, X, y)

        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        second_order_penalty = second_order_penalty.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)

        denominator = np.sum(
            (self.multipliers[0, 0] * sample_weight * (y - residual) * (1 - y + residual)) + second_order_penalty)

        # prevents overflow and division by zero
        if abs(denominator) < 1e-10:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = (numerator / denominator)


class ProxyLagrangianLeastSquaresError(LeastSquaresError):
    """Loss function for least squares (LS) estimation.
    Terminal regions do not need to be updated for least squares.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    def __init__(self,
                 constraints=None,
                 multiplier_stepsize=0.01,
                 update_type='multiplicative',
                 multipliers_radius=1.):
        super().__init__()

        # when the constraint is None, the problem is unconstrained
        if constraints is None:
            pass
        # otherwise, we consider a list of all constraints
        else:
            if not isinstance(constraints, list):
                self.constraints = [constraints]
            else:
                self.constraints = constraints

        self.multiplier_stepsize = multiplier_stepsize
        self.update_type = update_type
        self.multipliers_radius = multipliers_radius

        # number of constraints
        self.nb_constraints = len(self.constraints)

        # we start with zero for Lagrange multipliers
        # but one for the objective multiplier.
        self.multipliers = np.zeros((self.nb_constraints + 1, 1))
        self.multipliers[0, 0] = 1

    def _gradient_wrt_multiplier(self, y, raw_predictions):
        """Computes gradients of the Lagrangian with respect to multipliers. """
        # zero for objective is padded
        gradient_wrt_multiplier = np.zeros((self.nb_constraints+1, 1))

        # for each constraint we compute the constraint violation
        for i in range(self.nb_constraints):
            gradient_wrt_multiplier[i+1, 0] = self.constraints[i](y, raw_predictions)
        return gradient_wrt_multiplier

    def _first_order_penalties(self, y, raw_predictions):
        """Computes the penalty term for negative gradient. """
        # For each constraint we compute the penalty term, then multiply it
        # by Lagrange multiplier.
        proxy_penalties = np.array(
            [self.constraints[i].first_penalty(y, raw_predictions) for i in range(self.nb_constraints)])
        #  At the end, we sum up all penalties.
        sum_penalties = np.sum(np.multiply(self.multipliers[1:], proxy_penalties), axis=0)
        return sum_penalties

    def _update_multipliers(self, y, raw_predictions):
        """Updates the Lagrange multipliers and the multipliers distribution,
        Raises:
            ValueError: if update type is neither multiplicative or additive
        """
        # Computes the gradient of the Lagrangian wrt to the Lagrange multipliers
        gradient_wrt_multiplier = self._gradient_wrt_multiplier(y, raw_predictions)

        # If update type is multiplicative, updates the Lagrange multipliers
        # and the multipliers distribution
        self.multipliers = utils._multiplier_regular_update(
            self.multipliers,
            gradient_wrt_multiplier,
            self.multiplier_stepsize,
            self.multipliers_radius)

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples,)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        loss_gradient = self.multipliers[0, 0] * (
                y - raw_predictions.ravel())
        # Sum up the objectives and the constraints
        return loss_gradient + self._first_order_penalties(y, raw_predictions)

    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        """Least squares does not need to update terminal regions.
        But it has to update the predictions.
        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)
            The target labels.
        residual : ndarray of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n,)
            The weight of each sample.
        sample_mask : ndarray of shape (n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()
        self._update_multipliers(y, raw_predictions)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        pass