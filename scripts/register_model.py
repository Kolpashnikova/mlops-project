import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from datetime import datetime

from prefect import task

HPO_EXPERIMENT_NAME = "ridge-hyperopt"

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the experiment for autologging
mlflow.set_experiment(HPO_EXPERIMENT_NAME)
mlflow.sklearn.autolog()


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed data was saved"
)
@click.option(
    "--top_n",
    default=3,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
@task
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{HPO_EXPERIMENT_NAME}' not found.")
        return

    best_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=3,
        order_by=["metrics.test_rmse ASC"]
    )[0:top_n]

    # Define the stages for the models, with the first model being promoted to Production (best model)
    stages =["Production"] + ["Staging"] * (top_n - 1)

    # Register the best models
    for best_run in best_runs:
        print(f"run id: {best_run.info.run_id}, rmse: {best_run.data.metrics['rmse']:.4f}")
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_name = f"ridge-best-models"
        mlflow.register_model(model_uri, model_name)

        # Transition the model to the desired stage
        model_version = client.get_latest_versions(name=model_name)[0].version
        new_stage = stages.pop(0)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage,
            archive_existing_versions=True
        )

        # Update the model description
        date = datetime.today().date()
        client.update_model_version(
            name=model_name,
            version=model_version,
            description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
        )


if __name__ == '__main__':
    run_register_model()