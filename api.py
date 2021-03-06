# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from model import get_recommendations

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return "hello, world"

@app.route('/feed', methods=['POST'])
def feed():
        try:
            json_ = request.json

            recommendations_feed = get_recommendations(title=json_["title"])

            return recommendations_feed

        except:

            return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(port=12345, host='0.0.0.0', debug=True)
