from flask import Flask, render_template, request
import numpy as np
import joblib
import os
os.environ['IF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow
tf.config.threading.set_inter_op_parallerism_threads(1)
tf.config.threading.set_intra_op_parallerism_threads(1)

import keras

app = Flask(__name__)

# Load model and scaler
model = keras.models.load_model("breast_cancer_model.keras")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    probability = ""

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
            prob = model(input_scaled, training=False).numpy()[0][0]

            # Apply threshold
            if prob >= 0.5:
                prediction = "Malignant (Cancerous)"
            else:
                prediction = "Benign (Non-cancerous)"

            probability = f"{prob:.2%}"

        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = "N/A"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)