import mlflow
import os

# Set tracking URI to DagsHub
mlflow.set_tracking_uri('https://dagshub.com/raflialfansyah/Bank-Churn-MLflow-msml.mlflow')

# Ambil experiment ID
experiment = mlflow.get_experiment_by_name("Deep_Learning_Churn_Tuning")
if experiment is None:
    raise ValueError("Experiment not found di DagsHub!")

# Ambil run terakhir yang sukses
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id], 
    order_by=["start_time DESC"], 
    max_results=1
)

if runs.empty:
    raise ValueError("Tidak ada runs yang ditemukan di DagsHub!")

latest_run_id = runs.iloc[0].run_id
print(f"Berhasil menemukan Run ID terakhir: {latest_run_id}")

# Tulis Run ID ke GitHub Environment Variables agar bisa dipakai di step selanjutnya
env_file = os.getenv('GITHUB_ENV')
if env_file:
    with open(env_file, "a") as f:
        f.write(f"RUN_ID={latest_run_id}\n")
