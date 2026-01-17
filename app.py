from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow import keras

app = Flask(__name__)

# Load model and scaler
model = keras.models.load_model("breast_cancer_model.keras")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            # Get form inputs
            features = [
                float(request.form["clump_thickness"]),
                float(request.form["uniform_cell_size"]),
                float(request.form["uniform_cell_shape"]),
                float(request.form["marginal_adhesion"]),
                float(request.form["single_epithelial_size"]),
                float(request.form["bare_nuclei"]),
                float(request.form["bland_chromatin"]),
                float(request.form["normal_nucleoli"]),
                float(request.form["mitoses"]),
            ]

            # Convert to numpy array
            input_data = np.array([features])

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prob = model.predict(input_scaled)[0][0]

            # Apply threshold
            if prob >= 0.5:
                prediction = "Malignant (Cancerous)"
                label = 4
            else:
                prediction = "Benign (Non-cancerous)"
                label = 2

            probability = f"{prob:.2f}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)