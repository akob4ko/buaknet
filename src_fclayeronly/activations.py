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


class ReLU(object):
    @staticmethod
    def activate(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def activate_prime(z):
        primez = np.zeros_like(z)
        for j, value in np.ndenumerate(z):
            if value > 0.0:
                primez[j] = 1
        return primez


class Sigmoid(object):
    @staticmethod
    def activate(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def activate_prime(z):
        return Sigmoid.activate(z) * (1 - Sigmoid.activate(z))


class Softmax(object):
    @staticmethod
    def activate(z):
        out = np.zeros_like(z)
        if len(z.shape) > 2:
            for j in range(z.shape[0]):
                e_z = np.exp(z[j] - z[j].max(axis=0))
                out[j] = e_z / np.sum(e_z, axis=0)
        else:
            e_z = np.exp(z - z.max(axis=0))
            out = e_z / np.sum(e_z, axis=0)
        return out
