import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

# Define the directory name
directory = "data"

# Check if the directory exists
if not os.path.exists(directory):
    # If it does not exist, create it
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")

# Create a session with anonymous access
s3 = boto3.client('s3',
        region_name='nyc3',
        endpoint_url='https://nyc3.digitaloceanspaces.com',
        config=Config(signature_version=UNSIGNED))

# List objects in the bucket
bucket_name = 'mlops-project'
response = s3.list_objects_v2(Bucket=bucket_name)

for obj in response.get('Contents', []):
    print(obj['Key'])

# Download an object
object_key = 'data/atus37.parquet'
s3.download_file(bucket_name, object_key, object_key)
