# buaknet

**`buaknet`** is a convolutional neural network. In this case I use it for recognize handwritten digits.**`buaknet`** is written in Python. I've also added an app for test it with.

The front end shows a html canvas where you are asked to draw a digit, then it recognizes the handwritten digit using Machine Learning (ML) and outputs its prediction.

The ML model is a trained on the [MNIST](http://yann.lecun.com/exdb/mnist/)(database).

Give it a try at [buaknet.appspot.com](http://buaknet.appspot.com).

## Try it locally

Steps to download the source code and run Flask's development server locally.

1. Clone the repo and go inside
```shell
git clone https://github.com/akob4ko/buaknet.git
cd buaknet/
```
2. Create and activate a virtual environment
```shell
virtualenv venv
source venv/Scripts/activate
```

3. Install requirements 
```shell
pip install -r requirements.txt
```

4. Run app locally with Flask's development server
```shell
cd src
python digit_app.py

```
Go to address `http://127.0.0.1:5000/` on your web browser to use the app.


## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) file for details.

