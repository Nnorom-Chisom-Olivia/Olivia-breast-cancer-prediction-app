from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import tensorflow as tf

# Disable annoying TensorFlow logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# --- LOAD ASSETS (Requirement Part B.1) ---
# We look inside the /model/ folder for our saved files
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'breast_cancer_model.keras')
scaler_path = os.path.join(base_dir, 'model', 'scaler.pkl')

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        try:
            # These names MUST match the 'name=' attribute in your index.html
            features = [
                float(request.form["radius_mean"]),
                float(request.form["texture_mean"]),
                float(request.form["perimeter_mean"]),
                float(request.form["area_mean"]),
                float(request.form["smoothness_mean"])
            ]

            input_data = np.array([features])
            input_scaled = scaler.transform(input_data)

            # Use model.predict for Keras 3.x
            prob = model.predict(input_scaled)[0][0]
            
            prediction = "Malignant" if prob >= 0.5 else "Benign"

        except Exception as e:
            # This will show you exactly what is wrong on the webpage
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)