import random
import io
import os
import json
import conv2d_functions as c_funcs
import normalizations as norms
import activations as acts
import buaknet_preprocess as prep
import costs
import sys
from abc import ABC, abstractmethod
import numpy as np
import time


class Layer(ABC):
    def __init__(self, activation_fn=acts.Linear):
        self.weights = np.ndarray(())
        self.biases = np.ndarray(())
        self.minibatch_z = np.ndarray(())
        self.activation_fn = activation_fn
        self.minibatch_output = np.ndarray(())

    def save_params(self, network_name, layer_name, weights, biases, gamma, beta, mean, var):
        current_dir = os.getcwd()
        network_dir = current_dir + "/" + network_name
        if not os.path.exists(network_dir):
            os.mkdir(network_dir)
        os.chdir(network_dir)
        self.save_file(network_name, layer_name, weights, "weights")
        self.save_file(network_name, layer_name, biases, "biases")
        self.save_file(network_name, layer_name, gamma, "gamma")
        self.save_file(network_name, layer_name, beta, "beta")
        self.save_file(network_name, layer_name, mean, "mean")
        self.save_file(network_name, layer_name, var, "var")
        os.chdir(current_dir)

    def save_file(self, network_name, layer_name, param, param_name):
        file_streams = io.StringIO(network_name)
        np.savetxt(file_streams, param.reshape(-1, param.shape[-1]), delimiter=',')
        file_data = {param_name: file_streams.getvalue()}
        filename = network_name + '_' + layer_name + '-' + param_name+'.txt'
        if os.path.exists(filename):
            os.remove(filename)
        file = open(filename, 'w')
        json.dump(file_data, file)
        file.close()

    def load_params(self, network_name, layer_name, w_shape, b_shape, g_shape, beta_shape, mean_shape, var_shape):
        current_dir = os.getcwd()
        network_dir = current_dir + "//" + network_name
        os.chdir(network_dir)
        weights = self.load_file(network_name, layer_name, "weights", w_shape)
        biases = self.load_file(network_name, layer_name, "biases", b_shape)
        gamma = self.load_file(network_name, layer_name, "gamma", g_shape)
        beta = self.load_file(network_name, layer_name, "beta", beta_shape)
        mean = self.load_file(network_name, layer_name, "mean", mean_shape)
        var = self.load_file(network_name, layer_name, "var", var_shape)
        os.chdir(current_dir)
        params = {'weights': weights, 'biases': biases, 'gamma': gamma, 'beta': beta, 'mean': mean, 'var': var}
        return params

    def load_file(self, network_name, layer_name, param_name, param_shape):
        with open(network_name+'_'+layer_name+"-"+param_name+".txt") as param_file:
            param_data = json.load(param_file)
        param_streams = io.StringIO(param_data[param_name])
        param = np.loadtxt(param_streams, delimiter=',').reshape(param_shape)
        return param

    def update_params(self, delta_nabla_w, delta_nabla_b, traning_number, minibatch_size, learning_rate, lmbda):
        self.weights = (1 - learning_rate * (lmbda / traning_number)) * self.weights - (
                learning_rate / minibatch_size) * delta_nabla_w
        self.biases = self.biases - (learning_rate / minibatch_size) * delta_nabla_b

    def activate(self, z):
        return self.activation_fn.activate(z)

    def activate_prime(self, z):
        return self.activation_fn.activate_prime(z)

    @abstractmethod
    def forward(self, inpt):
        pass

    @abstractmethod
    def forward_minibatch(self, minibatch):
        pass

    @abstractmethod
    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        pass


class InputLayer(Layer):
    def __init__(self, batchsize=10, in_channel=0.0, width=0.0, height=0.0, activation_fn=acts.Linear):
        super().__init__(activation_fn)
        self.batchsize = batchsize
        self.in_channel = in_channel
        self.width = width
        self.height = height

    def forward(self, inpt):
        return inpt.reshape(self.in_channel, self.width, self.height)

    def forward_minibatch(self, inpt):
        self.minibatch_z = inpt.reshape(self.batchsize, self.in_channel, self.width, self.height)
        self.minibatch_output = self.minibatch_z
        return self.minibatch_output

    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        return minibatch_delta


class FlattenLayer(Layer):
    def __init__(self, batchsize=.0, in_channel=.0, width=.0, height=.0, activation_fn=acts.Linear):
        if sys.version_info[0] > 2:
            super().__init__(activation_fn)
        else:
            super(Layer).__init__(activation_fn)
        self.batchsize = batchsize
        self.in_channel = in_channel
        self.width = width
        self.height = height

    def forward(self, inpt):
        self.in_channel, self.width, self.height = inpt.shape
        return inpt.reshape(self.in_channel * self.width * self.height, 1)

    def forward_minibatch(self, inpt):
        self.batchsize, self.in_channel, self.width, self.height = inpt.shape
        self.minibatch_z = inpt.reshape(self.batchsize, self.in_channel * self.width * self.height, 1)
        self.minibatch_output = self.minibatch_z
        return self.minibatch_output

    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        return minibatch_delta.reshape(self.batchsize, self.in_channel, self.width, self.height)


class ConvLayer(Layer):
    def __init__(self, image_shape, in_channel, out_channel, filter_size, padding_size=0, stride=1,
                 activation_fn=acts.ReLU, normalization=norms.Linear):
        super().__init__(activation_fn)
        self.in_row_size = image_shape[0]
        self.in_col_size = image_shape[1]
        self.filter_size = filter_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding_size = padding_size
        self.stride = stride
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / out_channel * filter_size * filter_size),
                                        size=(in_channel, out_channel, filter_size, filter_size))
        self.biases = np.zeros((out_channel))
        self.minibatch_z_stride = np.ndarray(())
        self.normalization = normalization
        self.batchnorm_params = {'gamma': np.ones((out_channel)), 'beta': np.zeros((out_channel)),
                                 'mean': np.zeros((out_channel, 1, 1)), 'var': np.ones((out_channel, 1, 1))}
        self.batchnorm_cache = {}

    def forward(self, inpt):
        inpt = c_funcs.padding(inpt, self.padding_size)
        z = np.zeros((self.out_channel,
                      1 + (self.in_row_size - self.filter_size + self.padding_size * 2) // self.stride,
                      1 + (self.in_col_size - self.filter_size + self.padding_size * 2) // self.stride))
        for o in range(self.out_channel):
            for i in range(self.in_channel):
                z[o] += c_funcs.conv2d(inpt[i], self.weights[i, o], self.stride)
            z[o] += self.biases[o]
        z = (z - self.batchnorm_params['mean']) / np.sqrt(self.batchnorm_params['var'] + 1e-8)
        output_shape = (self.out_channel, 1, 1)
        z = self.batchnorm_params['gamma'].reshape(output_shape) * z \
            + self.batchnorm_params['beta'].reshape(output_shape)
        output = self.activate(z)
        return output

    def forward_minibatch(self, minibatch):
        minibatch = c_funcs.padding(minibatch, self.padding_size)
        minibatch_z = self.conv2d_forward(minibatch)
        self.minibatch_z, self.batchnorm_cache = self.normalization.normalize_forward_conv(
                                            minibatch_z, self.batchnorm_params['gamma'], self.batchnorm_params['beta'])
        self.batchnorm_params['mean'] = .9 * self.batchnorm_params['mean'] + .1 * self.batchnorm_cache['mu']
        self.batchnorm_params['var'] = .9 * self.batchnorm_params['var'] + .1 * self.batchnorm_cache['var']
        self.minibatch_output = self.activate(self.minibatch_z)
        return self.minibatch_output

    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        minibatch_delta, dgamma, dbeta = self.normalization.normalize_backward_conv(self.batchnorm_cache,
                                                                                    minibatch_delta,
                                                                                    self.batchnorm_params['gamma'])
        self.batchnorm_params['gamma'] = (1-learning_rate * (lmbda/traning_number)) * self.batchnorm_params['gamma'] - \
                                         (learning_rate / minibatch_size) * dgamma
        self.batchnorm_params['beta'] = (1-learning_rate * (lmbda/traning_number)) * self.batchnorm_params['beta'] - \
                                        (learning_rate / minibatch_size) * dbeta
        delta_nabla_w = self.conv2d_backward_weights(previouslayer, minibatch_delta)
        delta_nabla_b = minibatch_delta.sum(axis=3).sum(axis=2).sum(axis=0)

        self.update_params(delta_nabla_w, delta_nabla_b, traning_number, minibatch_size, learning_rate, lmbda)

        prevlayer_delta = self.conv2d_backward_delta(previouslayer, minibatch_delta)
        return prevlayer_delta

    def conv2d_forward(self, minibatch):
        filter_size = self.filter_size
        minibatch_size = minibatch.shape[0]
        minibatch_z = np.zeros((minibatch_size, self.out_channel,
                                1 + (self.in_row_size + self.padding_size * 2 - filter_size) // self.stride,
                                1 + (self.in_col_size + self.padding_size * 2 - self.filter_size) // self.stride))
        self.minibatch_z_stride = np.zeros((minibatch_size, self.out_channel,
                                            1 + (self.in_row_size + self.padding_size * 2 - filter_size) // self.stride,
                                            1 + (self.in_col_size + self.padding_size * 2 - filter_size) // self.stride,
                                            filter_size, filter_size))

        for b in range(minibatch_size):
            for o in range(self.out_channel):
                for i in range(self.in_channel):
                    minibatch_z[b, o] += c_funcs.conv2d(minibatch[b, i], self.weights[i, o], self.stride)
                    self.minibatch_z_stride[b, o] = c_funcs.strided2d(minibatch[b, i], self.weights[i, o], self.stride)
                minibatch_z[b, o] += self.biases[o]
        return minibatch_z

    def conv2d_backward_weights(self, previouslayer, minibatch_delta):
        minibatch_size = minibatch_delta.shape[0]
        delta_nabla_w = np.zeros_like(self.weights)
        for b in range(minibatch_size):
            for i in range(self.in_channel):
                for o in range(self.out_channel):
                    if hasattr(previouslayer, 'minibatch_z_stride') and self.stride > 1:
                        prev_z_180 = c_funcs.rot180(previouslayer.minibatch_z_stride[b, i])
                        prevz = previouslayer.activate(prev_z_180)
                        for y in range(self.minibatch_z_stride.shape[2]):
                            for x in range(self.minibatch_z_stride.shape[3]):
                                delta_nabla_w[i, o] += prevz[y, x] * minibatch_delta[b, o, y, x]
                    else:
                        prev_z_180 = c_funcs.rot180(previouslayer.minibatch_z[b, i])
                        prevz = previouslayer.activate(prev_z_180)
                        if self.stride > 1:
                            for y in range(self.minibatch_z_stride.shape[2]):
                                for x in range(self.minibatch_z_stride.shape[3]):
                                    delta_nabla_w[i, o] += prevz[y, x] * minibatch_delta[b, o, y, x]
                        else:
                            if self.padding_size > 0:
                                delta_nabla_w[i, o] += c_funcs.conv2d(prevz, c_funcs.condense(minibatch_delta[b, o],
                                                                                              self.padding_size),
                                                                      stride=1)
                            else:
                                delta_nabla_w[i, o] += c_funcs.conv2d(prevz, minibatch_delta[b, o], stride=1)

        return delta_nabla_w

    def conv2d_backward_delta(self, previouslayer, minibatch_delta):
        minibatch_size = minibatch_delta.shape[0]
        prevlayer_delta = np.zeros_like(previouslayer.minibatch_z)
        self_activation_prime = self.activate_prime(previouslayer.minibatch_z)
        for b in range(minibatch_size):
            for i in range(self.in_channel):
                for o in range(self.out_channel):
                    if self.stride > 1:
                        prevlayer_delta_padded = c_funcs.padding(c_funcs.stride_padding(minibatch_delta[b, o],
                                                                                        self.stride),
                                                                 (self.filter_size - 1) - self.padding_size)
                        prevlayer_delta[b, i] += c_funcs.conv2d(prevlayer_delta_padded,
                                                                c_funcs.rot180(self.weights)[i, o], 1)
                    elif previouslayer.__class__.__name__ != 'InputLayer':
                        if self.padding_size > 0:
                            padded_minibatch_delta = c_funcs.padding(minibatch_delta,
                                                                     self.filter_size - 1 - self.padding_size)
                        else:
                            padded_minibatch_delta = c_funcs.padding(minibatch_delta, self.filter_size - 1)

                        prevlayer_delta[b, i] += c_funcs.conv2d(padded_minibatch_delta[b, o],
                                                                c_funcs.rot180(self.weights)[i, o], stride=1)

        for b in range(minibatch_size):
            for i in range(self.in_channel):
                prevlayer_delta[b, i] *= self_activation_prime[b, i]
        return prevlayer_delta


class MaxPoolingLayer(Layer):
    def __init__(self, filter_size, activation_fn=acts.Linear):
        super().__init__(activation_fn)
        self.filter_size = filter_size
        self.offset_width = .0
        self.offset_height = .0
        self.flag = np.ndarray(())

    def forward(self, inpt):
        filter_size = self.filter_size
        in_channel, in_row, in_col = inpt.shape
        out_row_size = in_row // filter_size
        out_col_size = in_col // filter_size
        output = np.zeros((in_channel, out_row_size, out_col_size))
        for c in range(in_channel):
            for oy in range(out_row_size):
                for ox in range(out_col_size):
                    height = filter_size
                    width = filter_size
                    maxposition = np.argmax(
                        inpt[c, oy * filter_size: oy * filter_size + height,
                             ox * filter_size: ox * filter_size + width])
                    self.offset_height = int(maxposition // width)
                    self.offset_width = int(maxposition % width)
                    output[c, oy, ox] = inpt[c, oy * filter_size + self.offset_height,
                                             ox * filter_size + self.offset_width]
        return output

    def forward_minibatch(self, minibatch_inpt):
        filter_size = self.filter_size
        batchsize, in_channel, in_row, in_col = minibatch_inpt.shape
        out_row_size = in_row // filter_size
        out_col_size = in_col // filter_size
        self.flag = np.zeros_like(minibatch_inpt)
        minibatch_output = np.zeros((batchsize, in_channel, out_row_size, out_col_size))
        for b in range(batchsize):
            for c in range(in_channel):
                for oy in range(out_row_size):
                    for ox in range(out_col_size):
                        height = filter_size
                        width = filter_size
                        maxposition = np.argmax(
                            minibatch_inpt[b, c, oy * filter_size: oy * filter_size + height,
                                           ox * filter_size: ox * filter_size + width])
                        self.offset_height = int(maxposition // height)
                        self.offset_width = int(maxposition % width)
                        self.flag[b, c, oy * filter_size + self.offset_height,
                                  ox * filter_size + self.offset_width] = 1.0
                        minibatch_output[b, c, oy, ox] = minibatch_inpt[b, c, oy * filter_size + self.offset_height,
                                                                        ox * filter_size + self.offset_width]
        self.minibatch_z = minibatch_output
        self.minibatch_output = minibatch_output
        return self.minibatch_output

    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        filter_size = self.filter_size
        batchsize, in_channel, in_row, in_col = self.flag.shape
        out_row, out_col = minibatch_delta.shape[2:4]
        prevlayer_delta = np.zeros_like(self.flag)
        for b in range(batchsize):
            for c in range(in_channel):
                for oy in range(out_row):
                    for ox in range(out_col):
                        height = filter_size
                        width = filter_size
                        maxposition = np.argmax(
                            self.flag[b, c, oy * filter_size: oy * filter_size + height,
                                      ox * filter_size: ox * filter_size + width])
                        self.offset_height = int(maxposition // height)
                        self.offset_width = int(maxposition % width)
                        prevlayer_delta[b, c, oy * filter_size + self.offset_height,
                                        ox * filter_size + self.offset_height] = minibatch_delta[b, c, oy, ox]
        return prevlayer_delta


class FullyConnectedLayer(Layer):
    def __init__(self, input_number, output_number, activation_fn=acts.Sigmoid, normalization=norms.Linear):
        super().__init__(activation_fn)
        self.output_number = output_number
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(1.0/output_number),
                                        size=(output_number, input_number))
        self.biases = np.random.randn(output_number, 1)
        self.minibatch_z = np.ndarray(())
        self.normalization = normalization
        self.batchnorm_params = {'gamma': np.ones((output_number, 1)), 'beta': np.zeros((output_number, 1)),
                                 'mean': np.zeros((output_number, 1)), 'var': np.ones((output_number, 1))}
        self.batchnorm_cache = {}
        self.activation_fn = activation_fn
        self.minibatch_output = np.ndarray(())

    def forward(self, inpt):
        z = np.dot(self.weights, inpt) + self.biases
        z = (z - self.batchnorm_params['mean']) / np.sqrt(self.batchnorm_params['var'] + 1e-8)
        z = self.batchnorm_params['gamma'] * z + self.batchnorm_params['beta']
        output = self.activate(z)
        return output

    def forward_minibatch(self, minibatch):
        minibatch_z = np.matmul(self.weights, minibatch) + self.biases
        self.minibatch_z, self.batchnorm_cache = self.normalization.normalize_forward(
                                            minibatch_z, self.batchnorm_params['gamma'], self.batchnorm_params['beta'])
        self.batchnorm_params['mean'] = .9 * self.batchnorm_params['mean'] + .1 * self.batchnorm_cache['mu']
        self.batchnorm_params['var'] = .9 * self.batchnorm_params['var'] + .1 * self.batchnorm_cache['var']
        self.minibatch_output = self.activate(self.minibatch_z)
        return self.minibatch_output

    def outputerror(self, previouslayer, minibatch, traning_number, minibatch_size, learning_rate, lmbda, cost):
        errors = np.zeros((0, self.output_number, 1))
        for x, y in minibatch:
            error = cost.delta(self.minibatch_z, x, y)
            errors = np.append(errors, error.reshape((1, self.output_number, 1)), axis=0)

        delta_nabla_w = np.sum(np.matmul(errors, norms.batch_transpose(previouslayer.minibatch_output), axis=0))
        delta_nabla_b = np.sum(errors, axis=0)

        self.update_params(delta_nabla_w, delta_nabla_b, traning_number, minibatch_size, learning_rate, lmbda)

        prevz = previouslayer.minibatch_z
        prevlayer_activation_prime = previouslayer.activate_prime(prevz)

        prevlayer_delta = np.matmul(self.weights.transpose(), errors) * prevlayer_activation_prime
        return prevlayer_delta

    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        minibatch_delta, dgamma, dbeta = self.normalization.normalize_backward(self.batchnorm_cache, minibatch_delta,
                                                                               self.batchnorm_params['gamma'])

        self.batchnorm_params['gamma'] = (1-learning_rate * (lmbda/traning_number)) * self.batchnorm_params['gamma'] - \
                                         (learning_rate / minibatch_size) * dgamma
        self.batchnorm_params['beta'] = (1-learning_rate * (lmbda/traning_number)) * self.batchnorm_params['beta'] - \
                                        (learning_rate / minibatch_size) * dbeta

        delta_nabla_b = np.sum(minibatch_delta, axis=0)
        delta_nabla_w = np.sum(np.matmul(minibatch_delta,
                                         norms.batch_transpose(previouslayer.minibatch_output)), axis=0)

        self.update_params(delta_nabla_w, delta_nabla_b, traning_number, minibatch_size, learning_rate, lmbda)

        prevz = previouslayer.minibatch_z

        prevlayer_activation_prime = previouslayer.activate_prime(prevz)
        prevlayer_delta = np.matmul(self.weights.transpose(), minibatch_delta) * prevlayer_activation_prime
        return prevlayer_delta


class SoftmaxLayer(Layer):
    def __init__(self, input_number, output_number, activation_fn=acts.Softmax):
        super().__init__(activation_fn)
        self.output_number = output_number
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(1.0/output_number), size=(output_number, input_number))
        self.biases = np.random.randn(output_number, 1)

    def forward(self, inpt):
        z = np.dot(self.weights, inpt) + self.biases
        output = self.activate(z)
        return output

    def forward_minibatch(self, minibatch):
        self.minibatch_z = np.matmul(self.weights, minibatch) + self.biases
        self.minibatch_output = self.activate(self.minibatch_z)
        return self.minibatch_output

    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        raise NotImplementedError

    def outputerror(self, previouslayer, minibatch, traning_number, minibatch_size, learning_rate, lmbda, cost):
        errors = np.zeros((0, self.output_number, 1))
        for x, y in minibatch:
            error = cost.delta(self.minibatch_z, x, y)
            errors = np.append(errors, error.reshape((1, self.output_number, 1)), axis=0)

        delta_nabla_w = np.sum(np.matmul(errors, norms.batch_transpose(previouslayer.minibatch_output)), axis=0)
        delta_nabla_b = np.sum(errors, axis=0)

        self.update_params(delta_nabla_w, delta_nabla_b, traning_number, minibatch_size, learning_rate, lmbda)

        prevz = previouslayer.minibatch_z
        prevlayer_activation_prime = previouslayer.activate_prime(prevz)

        prevlayer_delta = np.matmul(self.weights.transpose(), errors) * prevlayer_activation_prime
        return prevlayer_delta


class Network(object):

    def __init__(self, name):
        self.name = name
        self.layers = []
        np.random.seed(1)

    def addlayer(self, layer):
        self.layers.append(layer)

    def save_params(self):
        for index, layer in enumerate(self.layers):
            if len(layer.weights.shape) > 0:  # weights has shape
                if hasattr(layer, 'batchnorm_params'):
                    layer.save_params(self.name, layer.__class__.__name__ + "-" + str(index), layer.weights,
                                      layer.biases,
                                      layer.batchnorm_params['gamma'], layer.batchnorm_params['beta'],
                                      layer.batchnorm_params['mean'], layer.batchnorm_params['var'])
                else:
                    layer.save_params(self.name, layer.__class__.__name__ + "-" + str(index),
                                      layer.weights, layer.biases, np.ones((1)), np.zeros((1)),
                                      np.zeros((1)), np.ones((1)))

    def load_params(self):
        for index, layer in enumerate(self.layers):
            if len(layer.weights.shape) > 0:  # weights has shape
                if hasattr(layer, 'batchnorm_params'):
                    params = layer.load_params(self.name, layer.__class__.__name__ + "-" + str(index),
                                               layer.weights.shape, layer.biases.shape,
                                               layer.batchnorm_params['gamma'].shape,
                                               layer.batchnorm_params['beta'].shape,
                                               layer.batchnorm_params['mean'].shape,
                                               layer.batchnorm_params['var'].shape)

                    self.layers[index].batchnorm_params['gamma'] = params['gamma']
                    self.layers[index].batchnorm_params['beta'] = params['beta']
                    self.layers[index].batchnorm_params['mean'] = params['mean']
                    self.layers[index].batchnorm_params['var'] = params['var']
                else:
                    params = layer.load_params(self.name, layer.__class__.__name__ + "-" + str(index),
                                               layer.weights.shape, layer.biases.shape, (1), (1), (1), (1))
                self.layers[index].weights = params['weights']
                self.layers[index].biases = params['biases']
        print("Saved params loaded!")

    def feedforward(self, in_data):
        for i in range(len(self.layers)):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        return in_data

    def batch_feedforward(self, minibatch):
        image_size = minibatch[0][0].shape[0]
        channels_number = minibatch[0][0].shape[1]
        result_size = minibatch[0][1].shape[0]
        minibatch_x = np.zeros((0, image_size, channels_number))
        minibatch_y = np.zeros((0, result_size, 1))
        for x, y in minibatch:
            minibatch_x = np.append(minibatch_x, x.reshape((1, image_size, channels_number)), axis=0)
            minibatch_y = np.append(minibatch_y, y.reshape((1, result_size, 1)), axis=0)

        for i in range(len(self.layers)):
            minibatch_x = self.layers[i].forward_minibatch(minibatch_x)
        return list(zip(minibatch_x, minibatch_y))

    def vectorized_result(self, j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    def training(self, training_data, validation_data, epochs, minibatch_size=10, learning_rate=0.5, lmbda=0.0,
                 costfn=costs.CrossEntropy, want_save_params=False):
        for idx, res in enumerate(training_data):
            res = list(res)
            res[1] = self.vectorized_result(res[1])
            training_data[idx] = tuple(res)

        traning_number = len(training_data)
        layers_number = len(self.layers)
        print("Traning started...")
        for j in range(epochs):
            epoch_start_time = time.time()
            if j != 0:
                print("Traning continue..")
            sys.stdout.flush()
            random.Random(1).shuffle(training_data)
            mini_batches = [
                training_data[k:k + minibatch_size]
                for k in range(0, traning_number, minibatch_size)]
            batch_number = 0
            for minibatch in mini_batches:
                batch_number += 1

                out_minibatch = self.batch_feedforward(minibatch)

                minibatch_delta = self.layers[-1].outputerror(self.layers[-2], out_minibatch, traning_number,
                                                              minibatch_size, learning_rate, lmbda, costfn)
                for laynum in range(2, layers_number):
                    minibatch_delta = self.layers[-laynum].backward(minibatch_delta, self.layers[-laynum - 1],
                                                                    traning_number, minibatch_size, learning_rate,
                                                                    lmbda)

                if (batch_number * minibatch_size) % (len(training_data) // 10) == 0:
                    print("Epoch {0}, Training mini-batch number {1}".format(j + 1, batch_number * minibatch_size))
                    sys.stdout.flush()
            print("Epoch %s training complete" % (j + 1))
            sys.stdout.flush()
            cost = .0
            for x, y in training_data:
                out_data = self.feedforward(x)
                cost += costfn.fn(out_data, y) / len(training_data)
            weightscost = sum(np.linalg.norm(self.layers[i].weights if hasattr(self.layers[i], 'weights') else 0) ** 2
                              for i in range(0, layers_number))
            cost += 0.5 * (lmbda / len(training_data)) * weightscost
            print("Cost on training data: {}".format(cost))
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in training_data]

            accuracy = sum(int(x == y) for (x, y) in results)
            print("Accuracy on training data: {}/ {}".format(accuracy, traning_number))

            cost = .0
            for x, y in validation_data:
                y = self.vectorized_result(y)
                out_data = self.feedforward(x)
                cost += costfn.fn(out_data, y) / len(validation_data)
            weightscost = sum(np.linalg.norm(self.layers[i].weights if hasattr(self.layers[i], 'weights') else 0) ** 2
                              for i in range(0, layers_number))
            cost += 0.5 * (lmbda / len(validation_data)) * weightscost
            print("Cost on validation data: {}".format(cost))

            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in validation_data]
            accuracy = sum(int(x == y) for (x, y) in results)
            print("Accuracy on validation data: {} / {}".format(accuracy, len(validation_data)))
            if want_save_params:
                self.save_params()
                print("Params has been saved!")
            print("Epoch time:")
            prep.print_elapsed_time(epoch_start_time)
            sys.stdout.flush()

    def predicate_one(self, test_inpt):
        prediction = self.feedforward(test_inpt)
        result = np.argmax(prediction)
        result_prob = prediction[result]
        return result, result_prob

    def predicate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y)
                   for (x, y) in test_data]
        accuracy = sum(int(x == y) for (x, y) in results)
        print("Accuracy on test data: {} / {}".format(accuracy, len(test_data)))
