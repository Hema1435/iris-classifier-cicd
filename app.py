# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Load the trained model from the saved file
model = joblib.load("iris_model.pkl")

# Initialize the Flask application
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract JSON data from the incoming POST request
        data = request.get_json(force=True)

        # Validate the input
        if "features" not in data or len(data["features"]) != 4:
            return jsonify({"error": "Exactly 4 numerical features are required"}), 400

        # Convert features to NumPy array
        features = np.array(data["features"], dtype=float).reshape(1, -1)

        # Predict the class
        prediction = model.predict(features)[0]

        # Map prediction to species name
        classes = ["setosa", "versicolor", "virginica"]
        result = {"prediction": classes[prediction]}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
 app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))   