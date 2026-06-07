import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Sklearn_Churn_Basic_Model")

mlflow.sklearn.autolog()

# Load dataset hasil preprocessing
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'churn-dataset_preprocessing', 'dataset_clean.csv')

print(f"Memuat dataset dari: {dataset_path}")
df = pd.read_csv(dataset_path)

# Memisahkan fitur dan target
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun dan melatih model tanpa hyperparameter tuning
print("Melatih model RandomForest Classifier...")
with mlflow.start_run(run_name="Basic_Model_Training"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
