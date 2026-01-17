import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

#Load UCI dataset
#Attach a column_names to the downloaded dataset
column_names = ["id","clump_thickness", "cell_size", "cell_shape", "Marginal_adhesion", 
"Epitheliel_size", "bare_nuclei", "Chromatin", "Nucleoli", "Mitoses", "class"]

#load and print uCI dataset
df=pd.read_csv("breast-cancer-wisconsin.data",names=column_names)
print(df.head())
df.to_csv("raw_breast_cancer_csv", index=False)

#data preprocessing
df["bare_nuclei"] = df["bare_nuclei"].astype(str)
df["bare_nuclei"] = df["bare_nuclei"].replace("?", pd.NA)
df.dropna(inplace=True)
df = df.drop(columns=["id"])
df = df.apply(pd.to_numeric)
df["class"] = df["class"].map({2: 0, 4: 1})
print(df.info())

#split the data into features(X) and labels(Y)
X = df.drop(columns=["class"])
y = df["class"]
print(X.shape)
print(set(y))

#split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_test.shape, y_train.shape)
print(set(y_train))
print(set(y_test))
print(y_train[:10])

#data normalization to enable the neural network work efficiently
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#fit on training data and transform both sets
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[:7])
print(X_test[:7])

#Build the ANN model
from tensorflow import keras
model = keras.Sequential([
keras.Input(shape=(9,)),
keras.layers.Dense(16, activation="relu"),
keras.layers.Dense(8, activation="relu"),
keras.layers.Dense(4, activation="relu"),  # New layer added
keras.layers.Dense(1, activation="sigmoid") # output layer
])

#compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), 
    tf.keras.metrics.Recall(name="recall")])

#Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test))

#Evaluate performance
results = model.evaluate(X_test, y_test, verbose = 0)
print("Model Evaluation Metrics") 
print(f"loss: {results[0]: .4f}")
print(f"Accuracy: {results[1]: .4f}")
print(f"precision: {results[2]: .4f}")
print(f"recall: {results[3]: .4f}")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("accuracy_plot.png")
plt.show()
plt.close()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("loss_plot.png")
plt.show()
plt.close()

#confusion matrix plots
#generate predictions
y_pred = (model.predict(X_test) > 0.5) .astype(int)
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted") 
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
plt.close()

# Calculate the false positive rate, true positive rate, and thresholds
y_pred_prob = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the Area Under the ROC Curve (AUC-ROC)
auc = roc_auc_score(y_test, y_pred_prob)

#Plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig("Roc_curve.png")
plt.show()

#save the model
model.save("breast_cancer_model.keras")
print("model saved successfully")

#Load the trained model
#model = keras.models.load_model("path_to_save_directory")
from tensorflow.keras.models import load_model
model = keras.models.load_model("breast_cancer_model.keras")

# Use the trained model to make a prediction on the new data point.
new_data = np.array([[2917023, 4, 1, 1, 3, 2, 1, 3, 1]])  # Replace with your actual data point

#Scale the features of the new data point using the same scaler that was used during training
new_data_scaled = scaler.transform(new_data)

#Use the trained model to make a prediction on the new data point.
y_pred_prob = model.predict(new_data_scaled)
 
#Interpret the predicted probability or convert it into a class label based on a threshold
#(e.g., 0.5) to classify it as benign or malignant.
threshold = 0.5
y_pred_original = np.where(y_pred == 1, 4, 2)

#The 'y_pred' variable contains the predicted tumor label for the new data point.
print("Considering the features of this tumour, this model is predicting it to be a - ", y_pred_original[0][0
])

import joblib
joblib.dump(scaler, "scaler.pkl")

'''the model predicted a class label of 4, which corresponds to a malignant tumor in the 
original UCI Breast Cancer Wisconsin dataset. This is to say that the model considers the tumour likely to be cancerous. in other words, 
the algorithm has determined that the characteristics of this tumour align more closely with malignant cases in the dataset, and therefore
the patient will be considered at risk of breast cancer according to this prediction.'''