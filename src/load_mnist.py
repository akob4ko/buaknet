import pickle
import gzip
import numpy as np
import os

def load_data():
    current_path = os.getcwd()
    data_path = current_path + "//data"
    f = gzip.open(data_path + '//mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    training_data = list(zip(training_inputs, tr_d[1]))
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)
