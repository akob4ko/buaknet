import numpy as np


def batch_transpose(inpt):
    tple = (inpt.shape[0],) + (inpt.shape[:0:-1])
    output = np.zeros(tple)
    for j in range(inpt.shape[0]):
        output[j] = inpt[j].transpose()
    return output


class Batchnorm(object):
    @staticmethod
    def normalize_forward(minibatch_z, gamma, beta):
        minibatch_z_withoutnorm = minibatch_z
        mu = np.sum(minibatch_z, axis=0) / minibatch_z.shape[0]

        var = np.sum((minibatch_z - mu) ** 2, axis=0) / minibatch_z.shape[0]

        norm = (minibatch_z - mu) / np.sqrt(var + 1e-8)
        minibatch_z_norm = gamma * norm + beta

        cache = {'z_withoutnorm': minibatch_z_withoutnorm, 'z_norm': minibatch_z_norm, 'mu': mu, 'var': var}
        return minibatch_z_norm, cache

    @staticmethod
    def normalize_forward_conv(minibatch_z, gamma, beta):
        minibatch_z_withoutnorm = minibatch_z
        batch_num, channels, height, width = minibatch_z.shape
        batch_size = batch_num * height * width

        mu = np.sum(minibatch_z, axis=(0, 2, 3), keepdims=True) / batch_size

        var = np.sum((minibatch_z - mu) ** 2, axis=(0, 2, 3), keepdims=True) / batch_size

        norm = (minibatch_z - mu) / np.sqrt(var + 1e-8)
        minibatch_z_norm = gamma.reshape(1, channels, 1, 1) * norm + beta.reshape(1, channels, 1, 1)
        cache = {'z_withoutnorm': minibatch_z_withoutnorm, 'z_norm': minibatch_z_norm,
                 'mu': mu.reshape(channels, 1, 1), 'var': var.reshape(channels, 1, 1)}
        return minibatch_z_norm, cache

    @staticmethod
    def normalize_backward(cache, minibatch_delta, gamma):
        minibatch_size = minibatch_delta.shape[0]
        z_subtrac_mu = cache['z_withoutnorm'] - cache['mu']
        std_inv = 1. / np.sqrt(cache['var'] + 1e-8)
        dminibatch_norm = minibatch_delta * gamma
        dvar = np.sum(dminibatch_norm * z_subtrac_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dminibatch_norm * -std_inv, axis=0) \
            + dvar * (np.sum(-2. * z_subtrac_mu, axis=0) / minibatch_size)

        dminibatch_z_norm = (dminibatch_norm * std_inv) + (dvar * 2 * z_subtrac_mu / minibatch_size) \
                                                        + (dmu / minibatch_size)
        dgamma = np.sum(minibatch_delta * cache['z_norm'], axis=0)
        dbeta = np.sum(minibatch_delta, axis=0)
        return dminibatch_z_norm, dgamma, dbeta

    @staticmethod
    def normalize_backward_conv(cache, minibatch_delta, gamma):
        batch_num, channels, height, width = minibatch_delta.shape
        batch_size = batch_num * height * width

        gamma = gamma.reshape(1, channels, 1, 1)

        z_subtrac_mu = cache['z_withoutnorm'] - cache['mu']
        std_inv = 1. / np.sqrt(cache['var'] + 1e-8)

        dminibatch_norm = minibatch_delta * gamma

        dvar = np.sum(dminibatch_norm * z_subtrac_mu, axis=(0, 2, 3), keepdims=True) * -.5 * std_inv ** 3
        dmu = np.sum(dminibatch_norm * -std_inv, axis=(0, 2, 3), keepdims=True) + dvar \
            * np.sum(-2 * z_subtrac_mu, axis=(0, 2, 3), keepdims=True) / batch_size

        dminibatch_z_norm = (dminibatch_norm * std_inv) + (dvar * 2 * z_subtrac_mu / batch_size) + (dmu / batch_size)

        dbeta = np.sum(minibatch_delta, axis=(0, 2, 3))
        dgamma = np.sum(minibatch_delta * cache['z_norm'], axis=(0, 2, 3))
        return dminibatch_z_norm, dgamma, dbeta


class Linear(object):
    @staticmethod
    def normalize_forward(minibatch_z, gamma, beta):
        cache = {'z_withoutnorm': 0, 'z': 0, 'mu': 0, 'var': 1}
        return minibatch_z, cache

    @staticmethod
    def normalize_backward(cache, minibatch_delta, gamma):
        dgamma = .0
        dbeta = .0
        return minibatch_delta, dgamma, dbeta

    @staticmethod
    def normalize_forward_conv(minibatch_z, gamma, beta):
        cache = {'z_withoutnorm': 0, 'z': 0, 'mu': 0, 'var': 1}
        return minibatch_z, cache

    @staticmethod
    def normalize_backward_conv(cache, minibatch_delta, gamma):
        dgamma = .0
        dbeta = .0
        return minibatch_delta, dgamma, dbeta
