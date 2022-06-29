from random import randint
from typing import Dict
import mlflow
import torch


def save_model(
    metric,
    artifact_file,
    model,
    sub_folder,
    tracking_uri,
    model_uri,
    scripted=False,
    parameters=None,
    tags={},
    experiment_name="",
    run_name="",
):
    """
    Store the results in airflow

    Args:
        metric:
        model:
        sub_folder: to identify the information about a model, site, ahead hours
        scripted: convert to jit or not
        artifact_file: path of the artifact file
        parameters:
        tracking_uri:
        model_uri:
        experiment_name:
        tags:{k:v}
        experiment_name:
        run_name:

    Returns:

    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(model_uri)
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name, artifact_location=tracking_uri)
    mlflow.set_experiment(experiment_name)
    eid = mlflow.get_experiment_by_name(experiment_name).experiment_id
    with mlflow.start_run(experiment_id=eid, run_name=run_name):
        assert isinstance(metric, Dict)
        mlflow.pytorch.autolog(log_models=False)
        mlflow.set_tags({**tags, "Mode": "best-setting"})
        mlflow.log_metrics(metric)
        mlflow.log_artifact(local_path=artifact_file)
        if scripted:
            model = torch.jit.script(model)
        mlflow.pytorch.log_model(model, artifact_path=sub_folder)


def load_model(uri):
    return mlflow.pytorch.load_model(uri)


if __name__ == "__main__":
    # uri = airflow.get_tracking_uri()

    m = load_model(
        "file:///Users/limingzhou/zhoul/work/me/xaog_proj/src/python/pytools/pytools/modeling/mlruns/0/52e8c03712764b88b78b6afc4f2843f4/artifacts/model"
    )
    pass
