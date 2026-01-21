import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# --- PART A.1: Load Dataset ---
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'raw_breast_cancer_csv')
data = pd.read_csv(csv_path)

# --- THE FIX: Rename columns to match Scorac requirements ---
# Mapping your existing data columns to the permitted features list
column_mapping = {
    'clump_thickness': 'radius_mean',
    'cell_size': 'texture_mean',
    'cell_shape': 'perimeter_mean',
    'Marginal_adhesion': 'area_mean',
    'Epitheliel_size': 'smoothness_mean',
    'class': 'diagnosis'
}
data = data.rename(columns=column_mapping)

# --- PART A.2: Preprocessing ---
# 1. Handling missing values (Requirement A.2.1)
# You dataset uses '?' for missing bare_nuclei, but since we are using 
# only the 5 features above, we just drop rows with nulls in those 5.
data = data.dropna(subset=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'])

# 2. Feature selection (Requirement A.2.2 - Picking exactly 5)
selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
X = data[selected_features]
y = data['diagnosis']

# 3. Encoding target (Requirement A.2.3)
# In your data, 2=Benign, 4=Malignant. We map them to 0 and 1.
y = y.map({2: 0, 4: 1})

# 4. Feature scaling (Requirement A.2.4 - Mandatory for Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- PART A.3 & A.4: Neural Network ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)

# --- 5. EVALUATE THE MODEL (Requirement Part A.5) ---
# Predict classes (0 for Benign, 1 for Malignant)
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate the 4 required metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("-" * 30)
print("SCORAC EVALUATION METRICS")
print("-" * 30)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("-" * 30)

# --- PART A.6: Save to /model/ folder ---
model.save(os.path.join(base_dir, 'breast_cancer_model.keras'))
joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))

print("\nDemonstrating Model Reloading...")
reloaded_model = tf.keras.models.load_model(os.path.join(base_dir, 'breast_cancer_model.keras'))
test_sample = X_test[:1]
demo_pred = reloaded_model.predict(test_sample)
print(f"Reloaded Model Prediction: {'Malignant' if demo_pred[0][0] > 0.5 else 'Benign'}")