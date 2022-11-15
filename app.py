from flask import Flask, request, jsonify
import pickle
from model import predict_resistance

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    yacht = request.get_json()

    with open('./model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    prediction = predict_resistance(yacht, model)

    result = {
        'resistance': prediction
    }
    return jsonify(result)
