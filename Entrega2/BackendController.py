from flask import Flask, request
import Entrega2.Pipeline as Pipeline

app = Flask(__name__)


@app.route("/predict")
def predict():
    pass