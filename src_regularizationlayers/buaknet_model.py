import buaknet
import buaknet_preprocess as prep
import activations as acts
import costs
import load_mnist
import time


def load_data():
    training_data, valid_data, test_data = load_mnist.load_data_wrapper()
    return training_data, valid_data, test_data


def build(has_saved_params):
    network = buaknet.Network("CNN1")
    network.addlayer(buaknet.InputLayer(10, 28, 28))
    network.addlayer(buaknet.FullyConnectedLayer(28*28, 100, activation_fn=acts.Sigmoid))
    network.addlayer(buaknet.SoftmaxLayer(100, 10))
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
    network.training(training_d[0:1000], valid_d[0:100], epochs=70, learning_rate=0.5, minibatch_size=10,lmbda=2.0,costfn=costs.CrossEntropy,
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
    train_main(False, True)
    evaluate_main()

