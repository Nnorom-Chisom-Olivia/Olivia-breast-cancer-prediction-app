from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import tensorflow as tf
import pandas as pd

# Optimization for Render Free Tier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# --- LOAD ASSETS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'breast_cancer_model.keras')
scaler_path = os.path.join(base_dir, 'model', 'scaler.pkl')

# Load model and scaler once when app starts
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        try:
            # 1. Capture inputs from the HTML form
            features_dict = {
                "radius_mean": [float(request.form["radius_mean"])],
                "texture_mean": [float(request.form["texture_mean"])],
                "perimeter_mean": [float(request.form["perimeter_mean"])],
                "area_mean": [float(request.form["area_mean"])],
                "smoothness_mean": [float(request.form["smoothness_mean"])]
            }

            # 2. Convert to DataFrame to keep feature names (Fixes the Warning)
            input_df = pd.DataFrame(features_dict)
            
            # 3. Scale and Predict
            input_scaled = scaler.transform(input_df)
            prob = model.predict(input_scaled)[0][0]
            
            # 4. Determine Result
            prediction = "Malignant" if prob >= 0.5 else "Benign"

        except Exception as e:
            # Display error message on the UI for debugging
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)