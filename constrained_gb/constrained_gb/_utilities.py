"""Contains functions for Lagrange multipliers updates.

We use two types of updates: multiplicative and additive. The multiplicative updates is used for
non-decomposable constraints, and the additive is for convex constraints. However, it practive
the additive type also works for non-decomposable constraints.
"""

import numpy as np


def _project_multipliers_wrt_euclidean_norm(multipliers, multipliers_radius=1.):
    """ Compute the Euclidean projection onto l1_norm ball
    Args:
        multipliers: 1d-array of shape (nb_constraint,)
            The Lagrangian multipliers of the constraints optimization problem
        multipliers_radius: float, optional, default: 1.,
            The radius of the Lagrangian multiplier space
    Returns:
        projected_multipliers: 1d-array of shape (nb_constraint,)
            Euclidean projection of Lagrangian multipliers onto l1-norm ball

    Notes:
        The code is forked from [1]

    References:
        [1] https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
    """
    assert multipliers_radius > 0., "Radius must be strictly positive (%r <= 0)" % multipliers_radius

    assert np.alltrue(multipliers >= 0.), "All multipliers must be positive (%r < 0)" % multipliers[multipliers < 0]

    # Check if we are already on the l1_ball
    if multipliers.sum() <= multipliers_radius:
        # Best projection: itself!
        return multipliers

    # Sort multipliers in ascending order
    sorted_multipliers = np.sort(multipliers)[::-1]

    # Number of multipliers
    n, = sorted_multipliers.shape

    # Computes cumulative sum of the multipliers
    cumulative_sum = np.cumsum(sorted_multipliers)

    # Finds rho
    selection = (sorted_multipliers * np.arange(1, n + 1) > (cumulative_sum - multipliers_radius))
    rho = np.argmax(np.arange(1, n + 1)[selection]) + 1

    # Compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cumulative_sum[rho - 1] - multipliers_radius) / rho

    # Compute the projection using theta
    projected_multipliers = (multipliers - theta).clip(min=0)
    return projected_multipliers


def _project_wrt_KL_divergence(multipliers_distribution):
    """ Project multipliers distribution with respect to KL divergence
        which is column-wise projection.
    Args:
        multipliers_distribution: 2d-array of shape (m+1, m+1)
           (m+1)-dimensional matrix, m is number of constraints
    """
    # Normalize the matrix column wise
    return multipliers_distribution / multipliers_distribution.sum(axis=0)

# def _power_method(A):
#     dimension = A.shape[0]
#
#     # Start the eigenvector with 1
#     old_eigenvector = np.ones(dimension) / dimension
#
#     done = True
#
#     while done:
#         # Computes multipliers_distribution * old_eigenvector
#         iteration = A.dot(old_eigenvector)
#
#         # Normalizes the mutiplication and takes it as updated eigenvector
#         new_eigenvector = iteration / np.sum(iteration)
#
#         # Normalizes the old eigenvector to compute the difference from new one
#         normalized_old_eigenvector = old_eigenvector / np.sum(old_eigenvector)
#
#         if (np.abs(normalized_old_eigenvector - new_eigenvector) < 0.0001).all():
#             # if two iterations differ by no more than epsilon, it terminates.
#             done = False
#         else:
#             # otherwise it updates the eigenvector
#             old_eigenvector = iteration
#     return new_eigenvector.reshape(-1, 1)

def _power_method(multipliers_distribution, epsilon=1e-2):
    dimension = multipliers_distribution.shape[1]

    # Start eigenvector
    old_eigenvector = np.ones(dimension) / np.sqrt(dimension)

    # Power iteration function
    def eigenvalue(matrix_, vector_):
        matrix_times_vector = matrix_.dot(vector_)
        return vector_.dot(matrix_times_vector)

    # Compute eigenvalue from iteration
    old_eigenvalue = eigenvalue(multipliers_distribution, old_eigenvector)

    while True:
        # Computes multipliers_distribution * old_eigenvector
        product = multipliers_distribution.dot(old_eigenvector)
        # Do one more iteration
        new_eigenvector = product / np.linalg.norm(product)

        # Update the eigenvalue
        new_eigenvalue = eigenvalue(multipliers_distribution, new_eigenvector)

        if np.abs(old_eigenvalue - new_eigenvalue) < epsilon:
            # if two iterations differ by no more than epsilon, it terminates.
            break
        old_eigenvector = new_eigenvector
        old_eigenvalue = new_eigenvalue

    return new_eigenvector.reshape(-1, 1)

def _multiplier_regular_update(old_multipliers,
                                gradient_wrt_multiplier,
                                multipier_stepsize=0.01,
                                multipliers_radius=1.0):
    """Updates Lagrange multipliers by gradient ascent in log-domain with small step-size.
        After update, it computes the Euclidean projection onto l1_norm ball.
    Args:
        old_multipliers: 2d-array of shape (m+1, 1).
            The Lagrange multipliers in the previous step, m is the number of constraints.
        gradient_wrt_multiplier: 2d-array of shape (m+1, 1).
            The gradient of the Lagrangian with respect to the multipliers which is the
            value of the constraints in the previous step.
        multipier_stepsize: float, optional, default = 0.01
            The learning rate to update Lagrangian multipliers by gradient ascend.
        multipliers_radius: float, optional, default = 1.0
            The radius of Lagrangian Multiplier space.
    Returns:
        updated_multipliers: 2d-array of shape (m+1, 1).
            The updated Lagrange multipliers.
    """
    new_multiplier = old_multipliers.copy()

    # If the value of the gradient wrt multiplier is negatives, it means the constraint
    # is not violated, so we do not penalize.
    new_multiplier[gradient_wrt_multiplier < 0] = 0.

    # Update and project the multipliers
    new_multiplier[1:, 0] = _project_multipliers_wrt_euclidean_norm(
        new_multiplier[1:, 0] + (multipier_stepsize * gradient_wrt_multiplier[1:, 0]), multipliers_radius)

    # Since, the first term of Lagrangian is not multiplied by a multiplier when the constraints are convex,
    # we set it to 1.
    new_multiplier[0, 0] = 1.
    return new_multiplier


def _multiplier_additive_update(old_multipliers,
                                gradient_wrt_multiplier,
                                multipier_stepsize=0.01,
                                multipliers_radius=1.0):
    """Updates Lagrange multipliers by gradient ascent in log-domain with small step-size.
        After update, it computes the Euclidean projection onto l1_norm ball.
    Args:
        old_multipliers: 2d-array of shape (m+1, 1).
            The Lagrange multipliers in the previous step, m is the number of constraints.
        gradient_wrt_multiplier: 2d-array of shape (m+1, 1).
            The gradient of the Lagrangian with respect to the multipliers which is the
            value of the constraints in the previous step.
        multipier_stepsize: float, optional, default = 0.01
            The learning rate to update Lagrangian multipliers by gradient ascend.
        multipliers_radius: float, optional, default = 1.0
            The radius of Lagrangian Multiplier space.
    Returns:
        updated_multipliers: 2d-array of shape (m+1, 1).
            The updated Lagrange multipliers.
    """
    new_multiplier = old_multipliers.copy()

    # If the value of the gradient wrt multiplier is negatives, it means the constraint
    # is not violated, so we do not penalize.
    new_multiplier[gradient_wrt_multiplier < 0] = 0.

    # Update and project the multipliers
    new_multiplier[1:, 0] = _project_multipliers_wrt_euclidean_norm(
        new_multiplier[1:, 0] + np.exp(multipier_stepsize * gradient_wrt_multiplier[1:, 0]), multipliers_radius)

    # Since, the first term of Lagrangian is not multiplied by a multiplier when the constraints are convex,
    # we set it to 1.
    new_multiplier[0, 0] = 1.
    return new_multiplier


def _multiplier_multiplicative_update(old_multipliers_distribution,
                                      old_multipliers,
                                      gradient_wrt_multipliers,
                                      multiplier_stepsize=0.01):
    """Updates Lagrange multipliers distribution multiplicative in log-domain with small step-size.
        After update, it projects the distribution wrt KL divergence.
        Then, the updated multipliers are the maximal eigenvector of the multipliers distribution,
        which is computed by power method.
    Args:
        old_multipliers_distribution: 2d-array of shape (m+1, m+1).
            The multipliers distribution in the previous step, m is the number of constraints.
        old_multipliers: 2d-array of shape (m+1, 1).
            The Lagrange multipliers in the previous step, m is the number of constraints.
        gradient_wrt_multiplier: 2d-array of shape (m+1, 1).
            The gradient of the Lagrangian with respect to the multipliers which is the
            value of the constraints in the previous step.
        multipier_stepsize: float, optional, default = 0.01
            The learning rate to update Lagrangian multipliers multiplicative.
    Returns:
        updated_multipliers_distribution: 2d-array of shape (m+1, m+1).
            The updated multipliers distribution.
        updated_multipliers: 2d-array of shape (m+1, 1).
            The updated Lagrange multipliers.
    """

    # Update multipliers in the log domain multiplicative.
    multiplicative_update = np.exp(
        multiplier_stepsize * np.multiply(
            gradient_wrt_multipliers.T, old_multipliers))[0,:]

    # Update and project multipliers distribution wrt KL divergence
    updated_multipliers_distribution = _project_wrt_KL_divergence(
        np.multiply(old_multipliers_distribution, multiplicative_update[:, None]))

    # Computes the maximal right-eigenvector of updated multipliers distribution
    updated_multipliers = _power_method(updated_multipliers_distribution)
    return updated_multipliers_distribution, updated_multipliers
