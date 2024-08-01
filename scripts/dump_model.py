import os
import pickle
import click
import mlflow
import boto3

from mlflow.tracking import MlflowClient

from prefect import flow, task

session = boto3.session.Session()
client_spaces = session.client('s3',
                        region_name='nyc3',
                        endpoint_url='https://nyc3.digitaloceanspaces.com',
                        aws_access_key_id=os.getenv('SPACES_KEY'),
                        aws_secret_access_key=os.getenv('SPACES_SECRET'))

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    

@click.command()
@click.option(
    "--dest_path",
    default="./model",
    help="Location where the resulting model will be saved"
)
@click.option(
    "--model_name",
    default="ridge-best-models",
    help="Model name"
)
@click.option(
    "--filename",
    default="model.pkl",
    help="Model filename"
)
@flow
def load_dump_model(model_name: str, dest_path: str, filename: str):
    client = MlflowClient()    
    latest_versions = client.get_latest_versions(name=model_name)

    for version in latest_versions:
        if version.current_stage == "Production":
            print(f"version: {version.version}, stage: {version.current_stage}")
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{version.version}")
            dump_pickle(model, os.path.join(dest_path, filename))
            print(f"Model saved to {os.path.join(dest_path, filename)}")

            client_spaces.put_object(Bucket='mlops-project', # The path to the directory you want to upload the object to, starting with your Space name.
                Key='model/model.pkl', # Object key, referenced whenever you want to access this file later.
                Body=open(os.path.join(dest_path, filename), 'rb'), # The file you want to upload.
                ACL='public-read' # Defines Access-control List (ACL) permissions, such as private or public.
                
            )
            return
        
    print("No model in Production stage found.")

if __name__ == '__main__':
    load_dump_model()
