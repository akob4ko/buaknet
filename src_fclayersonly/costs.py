import activations as acts
import numpy as np


class Quadratic(object):

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * acts.Sigmoid.activate_prime(z)


