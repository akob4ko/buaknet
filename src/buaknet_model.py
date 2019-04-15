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
    conv1 = buaknet.ConvLayer((28, 28), 1, 8, 5, stride=2, padding_size=0, normalization=norms.Batchnorm)
    conv2 = buaknet.ConvLayer((12, 12), 8, 16, 5, stride=2, padding_size=0, normalization=norms.Batchnorm)
    flatten1 = buaknet.FlattenLayer()
    fc1 = buaknet.FullyConnectedLayer(16 * 4 * 4, 100, activation_fn=acts.ReLU, normalization=norms.Batchnorm)
    fc2 = buaknet.FullyConnectedLayer(100, 100, activation_fn=acts.ReLU, normalization=norms.Batchnorm)
    sm = buaknet.SoftmaxLayer(100, 10)
    network.addlayer(il)
    network.addlayer(conv1)
    network.addlayer(conv2)
    network.addlayer(flatten1)
    network.addlayer(fc1)
    network.addlayer(fc2)
    network.addlayer(sm)
    if has_saved_params:
        network.load_params()
    return network


def train_main(has_saved_params=False, want_save_params=True):
    print('-------------------------------------')
    print('               BUAKNET')
    print('         MNIST classifier')
    print('-------------------------------------')

    print('Loading training and validation and test data...')
    training_d, valid_d, test_d = load_data()
    print('Data loaded!')
    network = build(has_saved_params)
    print('Network was built!')
    training_started_time = time.time()
    network.training(training_d, valid_d, 10, learning_rate=0.03, lmbda=2.0, minibatch_size=10,
                     want_save_params=want_save_params)
    print('Traning completed in!')
    prep.print_elapsed_time(training_started_time)


def evaluate_main():
    print('-------------------------------------')
    print('               BUAKNET')
    print('         MNIST classifier')
    print('-------------------------------------')

    print('Loading training, validation and test data...')
    training_d, valid_d, test_d = load_data()
    print('Data loaded!')
    network = build(True)
    print('Network was built!')
    print('Evaluation started...')
    network.predicate(test_d)
    print('Evaluation done!')


if __name__ == '__main__':
    train_main(False, True)
    evaluate_main()
