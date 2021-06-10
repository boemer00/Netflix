import mlflow
from mlflow.tracking import MlflowClient
EXPERIMENT_NAME = "Pris_LND_Netflix_experiment"
mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

for model in ["linear", "Randomforest"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 2)
    client.log_param(run.info.run_id, "test", model)
    client.log_param(run.info.run_id, "student_name", "Priscilla")
