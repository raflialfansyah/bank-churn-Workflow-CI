import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import mlflow
import dagshub

os.environ['MLFLOW_TRACKING_USERNAME'] = 'raflialfansyah'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'c68b35604c3f56edcf130e7683248018c486cd0e' # Your DagsHub Token

mlflow.set_tracking_uri('https://dagshub.com/raflialfansyah/Bank-Churn-MLflow-msml.mlflow')

dagshub.init(repo_owner='raflialfansyah', repo_name='Bank-Churn-MLflow-msml', mlflow=True)

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'churn-dataset_preprocessing', 'dataset_clean.csv')
df = pd.read_csv(dataset_path)
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bangun model
def build_model(hp):
    model = keras.Sequential()
    # Tuning jumlah neuron pada layer pertama
    model.add(keras.layers.Dense(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        activation='relu', 
        input_shape=(X_train.shape[1],)
    ))
    # Tuning dropout
    model.add(keras.layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Tuning learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Hyperparameter tuning
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='tuner_logs',
    project_name='churn_tuning'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best units: {best_hps.get('units_1')}, Best LR: {best_hps.get('learning_rate')}")

# Training model dengan hyperparameter terbaik
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)

# Eval dan artefak
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# MLflow manual logging
mlflow.set_experiment("Deep_Learning_Churn_Tuning")

with mlflow.start_run():
    # Log hyperparameter
    mlflow.log_param("best_units_1", best_hps.get('units_1'))
    mlflow.log_param("best_learning_rate", best_hps.get('learning_rate'))
    mlflow.log_param("best_dropout", best_hps.get('dropout'))
    
    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_log_loss", log_loss(y_test, y_pred_prob))
    
    # Log artefak (Model & Plots)
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact('roc_curve.png')
    
    # Log tuner_logs
    mlflow.log_artifacts('tuner_logs', artifact_path='tuner_logs_history')
    
    # Log model keras 
    mlflow.keras.log_model(model, "keras_churn_model")
    print("Model and artifacts successfully saved to MLflow / DagsHub!")
