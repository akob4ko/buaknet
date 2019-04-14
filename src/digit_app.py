import buaknet_preprocess as prep
import buaknet_model as bamodel
import numpy as np
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
network = bamodel.build(True)


def preprocess(digit_base64_image):
    digit_img = prep.base64_str_to_numpy(digit_base64_image)
    digit = prep.resize_img(digit_img)
    digit = prep.reshape_array(digit)
    return digit


@app.route('/get_digit', methods=['POST'])
def get_digit():
    digit = request.data
    digit = preprocess(digit)
    prediction, result_prob = network.predicate_one(digit)
    prob = np.around(result_prob, decimals=3) * 100
    if prob < 60:
        prediction = 'Hm...'
        probability = 'Do you call that a digit?'
    else:
        prediction = str(prediction)
        probability = '({}% probability)'.format(str(int(prob)))

    return jsonify(result=prediction, probability=probability)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
