import buaknet
import buaknet_preprocess as prep
import normalizations as norms
import activations as acts
import load_mnist
import time


def load_data():
    training_data, valid_data, test_data = load_mnist.load_data_wrapper()
    return training_data, valid_data, test_data


def build(has_saved_params):
    network = buaknet.Network("CNN1")
    il = buaknet.InputLayer(10, 1, 28, 28)
    conv1 = buaknet.ConvLayer((28, 28), 1, 20, 5, stride=1, padding_size=0, normalization=norms.Batchnorm)
    maxpool1 = buaknet.MaxPoolingLayer(2)
    flatten1 = buaknet.FlattenLayer()
    fc1 = buaknet.FullyConnectedLayer(20 * 12 * 12, 200, activation_fn=acts.ReLU, normalization=norms.Batchnorm)
    fc2 = buaknet.FullyConnectedLayer(200, 100, activation_fn=acts.ReLU, normalization=norms.Batchnorm)
    sm = buaknet.SoftmaxLayer(100, 10)
    network.addlayer(il)
    network.addlayer(conv1)
    network.addlayer(maxpool1)
    network.addlayer(flatten1)
    network.addlayer(fc1)
    network.addlayer(fc2)
    network.addlayer(sm)

    if has_saved_params:
        network.load_params()
    return network


def train_main(has_saved_params=False, want_save_params=True):
    print('-------------------------------------', flush=True)
    print('               BUAKNET', flush=True)
    print('         MNIST classifier', flush=True)
    print('-------------------------------------', flush=True)

    print('Loading training and validation and test data...', flush=True)
    training_d, valid_d, test_d = load_data()
    print('Data loaded!', flush=True)
    network = build(has_saved_params)
    print('Network was built!', flush=True)
    training_started_time = time.time()
    network.training(training_d, valid_d, 10, learning_rate=0.03, lmbda=0.0, minibatch_size=10,
                     want_save_params=want_save_params)
    print('Traning completed in!', flush=True)
    prep.print_elapsed_time(training_started_time)


def evaluate_main():
    print('-------------------------------------', flush=True)
    print('               BUAKNET', flush=True)
    print('         MNIST classifier', flush=True)
    print('-------------------------------------', flush=True)

    print('Loading training, validation and test data...', flush=True)
    training_d, valid_d, test_d = load_data()
    print('Data loaded!', flush=True)
    network = build(True)
    print('Network was built!', flush=True)
    print('Evaluation started...', flush=True)
    network.predicate(test_d)
    print('Evaluation done!', flush=True)


if __name__ == '__main__':
    # train_main(False, True)
    evaluate_main()
