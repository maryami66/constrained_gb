import numpy as np
import gradient_boosting_constrained_optimization._utilities as util



def test_1():
    nb_constraints = 3

    gradient_wrt_multipliers = [
        np.array([[0.0], [0.8]]),
        np.array([[0.0], [0.8]]),
        np.array([[0.0], [0.6]]),
        np.array([[0.0], [0.4]]),
        np.array([[0.0], [0.0]])]


    multiplier = np.array([[1.0], [0.0]])

    multipier_stepsize = 0.01

    multipliers_radius = 1.0

    for grad in gradient_wrt_multipliers:
        multiplier_update = util._multiplier_additive_update(multiplier, grad,
                                                             multipier_stepsize=multipier_stepsize,
                                                             multipliers_radius=multipliers_radius)
        multiplier = multiplier_update
        print(multiplier_update)

test_1()


# ### Test multipicative update
#
# multiplier = np.random.rand(nb_constraints+1).reshape(-1, 1)
#
# M = np.ones((nb_constraints + 1,
#              nb_constraints + 1)) / nb_constraints + 1
#
#
# print(M)
#
# print(multiplier)
#
# multiplier_update = util._multiplier_multiplicative_update(M,
#                                                            multiplier,
#                                                            gradient_wrt_multipliers,
#                                                            multiplier_stepsize=multipier_stepsize)
#
# print(multiplier_update)