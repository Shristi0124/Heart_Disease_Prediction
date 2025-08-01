import numpy as np
from flask import Flask, request, render_template
import pickle

# Create Flask app
app = Flask(__name__)

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    # Scale input
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
