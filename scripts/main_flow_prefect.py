from prefect import flow, task
import os
import pickle
from datetime import datetime
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from prefect import flow, task

HPO_EXPERIMENT_NAME = "ridge-hyperopt"
MODELNAME = f"ridge-best-models"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(HPO_EXPERIMENT_NAME)

client = MlflowClient()

@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    return df


@task
def prepare_data(data):
    target_column = 'earnweek'  
    data = data.dropna(subset=[target_column])
    X = data.drop(columns = target_column)  
    y = data[target_column]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

@task
def preprocess(X_train, X_val):
    numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Preprocessing pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    return X_train, X_val, preprocessor

@flow
def run_data_prep(raw_data_path = "./data", dest_path = "./output", dataset = "atus37.parquet"):
    # Load data file
    df = read_dataframe(os.path.join(raw_data_path, dataset))

    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data(df)


    # Preprocess data
    X_train, X_val, preprocessor = preprocess(X_train, X_val)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save preprocessor and datasets
    dump_pickle(preprocessor, os.path.join(dest_path, "preprocessor.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))

@flow
def run_training(data_path = "./output"):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    mlflow.sklearn.autolog()
    
    with mlflow.start_run():

        rf = LinearRegression()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metrics({'rmse': rmse})

        print(f"RMSE: {rmse}")

@flow
def run_register_model(data_path = "./output", top_n = 3):

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
        mlflow.register_model(model_uri, MODELNAME)

        # Transition the model to the desired stage
        model_version = client.get_latest_versions(name=MODELNAME)[0].version
        new_stage = stages.pop(0)
        client.transition_model_version_stage(
            name=MODELNAME,
            version=model_version,
            stage=new_stage,
            archive_existing_versions=False
        )

        # Update the model description
        date = datetime.today().date()
        client.update_model_version(
            name=MODELNAME,
            version=model_version,
            description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
        )


@flow
def run_load_dump_model(dest_path = "./model", filename = "model.pkl", modelname = MODELNAME):
    latest_versions = client.get_latest_versions(name=modelname)

    for version in latest_versions:
        if version.current_stage == "Production":
            print(f"version: {version.version}, stage: {version.current_stage}")
            model = mlflow.sklearn.load_model(f"models:/{modelname}/{version.version}")
            dump_pickle(model, os.path.join(dest_path, filename))
            print(f"Model saved to {os.path.join(dest_path, filename)}")
            return
        
    print("No model in Production stage found.")

@flow(name = "wages_prediction")
def run_script():
    run_data_prep()
    run_training()
    run_register_model()
    run_load_dump_model()

if __name__ == '__main__':
    run_script()

    