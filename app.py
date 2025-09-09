# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
with open("disease_model.pkl", "rb") as f:
    model, all_symptoms, disease_encoder = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        # Get 7 input symptoms
        inputs = [request.form.get(f"symptom{i}") for i in range(1, 8)]
        inputs = [s.lower().strip() for s in inputs if s]  # clean

        # Convert to vector
        vector = [1 if s in inputs else 0 for s in all_symptoms]
        X = np.array([vector])

        # Predict
        pred = model.predict(X)[0]
        prediction = disease_encoder.inverse_transform([pred])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
