import os
import click
import boto3

from prefect import flow, task

session = boto3.session.Session()
client_spaces = session.client('s3',
                        region_name='nyc3',
                        endpoint_url='https://nyc3.digitaloceanspaces.com',
                        aws_access_key_id=os.getenv('SPACES_KEY'),
                        aws_secret_access_key=os.getenv('SPACES_SECRET'))


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the pickles will be saved"
)
@click.option(
    "--filenames",
    default="preprocessor.pkl, train.pkl, val.pkl",
    help="Model filename"
)
@flow
def load_pickes(data_path: str, filenames: str):

    filenames = filenames.split(", ")

    for filename in filenames:
        client_spaces.put_object(Bucket='mlops-project', 
            Key=f'output/{filename}', 
            Body=open(os.path.join(data_path, filename), 'rb'), 
            ACL='public-read'            
        )
    return

if __name__ == '__main__':
    load_pickes()
