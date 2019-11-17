import random
import io
import os
import json
import activations as acts
import costs
import sys
from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self, activation_fn=acts.Linear):
        self.weights = np.ndarray(())
        self.biases = np.ndarray(())
        self.minibatch_z = np.ndarray(())
        self.activation_fn = activation_fn
        self.minibatch_output = np.ndarray(())

    def save_params(self, network_name, layer_name, weights, biases):
        current_dir = os.getcwd()
        network_dir = current_dir + "/" + network_name
        if not os.path.exists(network_dir):
            os.mkdir(network_dir)
        os.chdir(network_dir)
        self.save_file(network_name, layer_name, weights, "weights")
        self.save_file(network_name, layer_name, biases, "biases")
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

    def load_params(self, network_name, layer_name, w_shape, b_shape):
        current_dir = os.getcwd()
        print(current_dir)
        network_dir = current_dir + "/" + network_name
        os.chdir(network_dir)
        weights = self.load_file(network_name, layer_name, "weights", w_shape)
        biases = self.load_file(network_name, layer_name, "biases", b_shape)
        os.chdir(current_dir)
        params = {'weights': weights, 'biases': biases}
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
    def __init__(self, batchsize=10, width=0.0, height=0.0, activation_fn=acts.Linear):
        super().__init__(activation_fn)
        self.batchsize = batchsize
        self.width = width
        self.height = height

    def forward(self, inpt):
        return inpt.reshape(self.width* self.height,1)

    def forward_minibatch(self, inpt):
        self.minibatch_z = inpt.reshape(self.batchsize, self.width*self.height, 1)
        self.minibatch_output = self.minibatch_z
        return self.minibatch_output

    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        return minibatch_delta


class FullyConnectedLayer(Layer):
    def __init__(self, input_number, output_number, activation_fn=acts.Sigmoid):
        super().__init__(activation_fn)
        self.output_number = output_number
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(1.0/output_number),
                                        size=(output_number, input_number))
        # large weight initializer
        # self.weights = np.random.normal(loc=0.0, scale=np.sqrt(output_number), size=(output_number, input_number))
        self.biases = np.random.randn(output_number, 1)
        self.minibatch_z = np.ndarray(())
        self.activation_fn = activation_fn
        self.minibatch_output = np.ndarray(())

    def forward(self, inpt):
        z = np.dot(self.weights, inpt) + self.biases
        output = self.activate(z)
        return output

    def forward_minibatch(self, minibatch):
        self.minibatch_z = np.matmul(self.weights, minibatch) + self.biases
        self.minibatch_output = self.activate(self.minibatch_z)
        return self.minibatch_output

    def batch_transpose(self, inpt):
        tple = (inpt.shape[0],) + (inpt.shape[:0:-1])
        output = np.zeros(tple)
        for j in range(inpt.shape[0]):
            output[j] = inpt[j].transpose()
        return output

    def backward(self, minibatch_delta, previouslayer, traning_number, minibatch_size, learning_rate, lmbda):
        delta_nabla_b = np.sum(minibatch_delta, axis=0)
        delta_nabla_w = np.sum(np.matmul(minibatch_delta,
                                         self.batch_transpose(
                                             previouslayer.minibatch_output)), axis=0)
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

    def batch_transpose(self, inpt):
        tple = (inpt.shape[0],) + (inpt.shape[:0:-1])
        output = np.zeros(tple)
        for j in range(inpt.shape[0]):
            output[j] = inpt[j].transpose()
        return output

    def outputerror(self, previouslayer, minibatch, traning_number, minibatch_size, learning_rate, lmbda, cost):
        errors = np.zeros((0, self.output_number, 1))
        for count, (x, y) in enumerate(minibatch):
            error = cost.delta(self.minibatch_z[count], x, y)
            errors = np.append(errors, error.reshape((1, self.output_number, 1)), axis=0)
        delta_nabla_w = np.sum(np.matmul(errors, self.batch_transpose(previouslayer.minibatch_output)), axis=0)
        delta_nabla_b = np.sum(errors, axis=0)

        self.update_params(delta_nabla_w, delta_nabla_b, traning_number, minibatch_size, learning_rate,lmbda)

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
                    layer.save_params(self.name, layer.__class__.__name__ + "-" + str(index),
                                      layer.weights, layer.biases)

    def load_params(self):
        for index, layer in enumerate(self.layers):
            if len(layer.weights.shape) > 0:  # weights has shape
                params = layer.load_params(self.name, layer.__class__.__name__ + "-" + str(index),
                                           layer.weights.shape, layer.biases.shape)
                self.layers[index].weights = params['weights']
                self.layers[index].biases = params['biases']
        print("Saved params loaded!", flush=True)

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

    def training(self, training_data, validation_data, epochs, minibatch_size=10, learning_rate=0.5,lmbda=0.0,
                 costfn=costs.CrossEntropy, want_save_params=False):

        traning_number = len(training_data)
        layers_number = len(self.layers)
        print("Traning started...", flush=True)
        sys.stdout.flush()
        for j in range(epochs):
            if j != 0:
                print("Traning continue..", flush=True)
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
                                                              minibatch_size, learning_rate,lmbda, costfn)
                for laynum in range(2, layers_number):
                    minibatch_delta = self.layers[-laynum].backward(minibatch_delta, self.layers[-laynum - 1],
                                                                    traning_number, minibatch_size, learning_rate,
                                                                    lmbda)

                if (batch_number * minibatch_size) % (len(training_data) // 10) == 0:
                    print("Epoch {0}, Training mini-batch number {1}".format(j + 1, batch_number * minibatch_size),
                          flush=True)
            sys.stdout.flush()

            print("Epoch %s training complete" % (j + 1), flush=True)
            sys.stdout.flush()

            cost = .0
            for x, y in training_data:
                out_data = self.feedforward(x)
                cost += costfn.fn(out_data, y) / len(training_data)
            print("Cost on training data: {}".format(cost), flush=True)
            sys.stdout.flush()

            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in training_data]

            accuracy = sum(int(x == y) for (x, y) in results)
            print("Accuracy on training data: {}/ {}".format(accuracy, traning_number), flush=True)
            sys.stdout.flush()

            cost = .0
            for x, y in validation_data:
                y = self.vectorized_result(y)
                out_data = self.feedforward(x)
                cost += costfn.fn(out_data, y) / len(validation_data)
            print("Cost on validation data: {}".format(cost), flush=True)
            sys.stdout.flush()

            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in validation_data]
            accuracy = sum(int(x == y) for (x, y) in results)
            print("Accuracy on validation data: {} / {}".format(accuracy, len(validation_data)), flush=True)
            if want_save_params:
                self.save_params()
                print("Params has been saved!", flush=True)
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
        print("Accuracy on test data: {} / {}".format(accuracy, len(test_data)), flush=True)
