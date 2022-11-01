from flask import Flask, request, jsonify
import pickle
from model import predict_resistance

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    yacht = request.get_json()

    prediction = predict_resistance(yacht, model)

    result = {
        'resistance': list(prediction)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
