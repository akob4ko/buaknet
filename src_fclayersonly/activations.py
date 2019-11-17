import numpy as np


class Linear(object):
    @staticmethod
    def activate(z):
        return z

    @staticmethod
    def activate_prime(z):
        primez = np.zeros_like(z)
        for j, value in np.ndenumerate(z):
            primez[j] = 1
        return primez


class Sigmoid(object):
    @staticmethod
    def activate(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def activate_prime(z):
        return Sigmoid.activate(z) * (1 - Sigmoid.activate(z))
