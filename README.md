# MLOps Project

# Problem Description: Wages Prediction Model Pipeline

## Objective
The goal of this project is to develop a machine learning pipeline that predicts weekly earnings (`earnweek`) based on various features from a dataset. The pipeline is orchestrated using Prefect and integrates several key technologies including Scikit-Learn for model training, MLflow for experiment tracking and model management, and DigitalOcean Spaces for data storage.

## Technologies and Tools

- **Scikit-Learn**: For data preprocessing, model training, and evaluation.
- **MLflow**: For experiment tracking, model management, and registry.
- **Prefect**: For orchestrating the entire machine learning workflow, managing tasks, and ensuring smooth execution of each step.
- **Boto3**: For interacting with DigitalOcean Spaces to download and upload data files.
- **Evidently AI**: For monitoring and analyzing machine learning models in production to ensure their performance and stability over time.
- **Pandas**: For data manipulation and preparation.
- **Flask**: For the model to be deployed as a web service.

## Pipeline Workflow

1. **Download Data Files**:
    - Retrieve data and preprocessed files from Digitalocean Spaces and save them locally.

2. **Prepare Data**:
    - Read the downloaded dataset, split it into features and target variables, and further split into training and validation sets.
    - Preprocess the data by handling missing values, scaling, and encoding features.

3. **Train Model**:
    - Train a Linear Regression model on the preprocessed training data.
    - Evaluate the model on the validation set and log the metrics using MLflow.

4. **Register Model**:
    - Identify the best model based on validation performance metrics.
    - Register and transition the best model to the appropriate stage in MLflow.

5. **Load and Save Model**:
    - Load the best model from MLflow (based on the production stage) and save it locally for deployment.





## Dockerization of the Model

You can try and run the model on your own machine (if you don't want to use the cloud version -- also the cloud version will not be available after this project is reviewed). Using the following, you can 1) run only the model in docker, 2) run the model and mlflow in docker.

### 1) Run the Dockerized Model

- build docker image from Dockerfile:

```bash
docker build -t wage-prediction:v1 .
```

- run the Docker image with the model:

```bash
docker run -it -p 9696:9696 wage-prediction:v1
```

- run test script to test if it's doing its job:

```bash
python scripts/test_prediction.py
```

### 2) Dockerized Version of the Model + Mlflow

This will copy the experiments that are already present into the docker container. If new experiments are run, then they will be in the docker container, not on the local machine.

- remove all containers and images
```bash
docker rm -vf $(docker ps -aq)
docker rmi -f $(docker images -aq)
```

- compose docker images
```bash
docker-compose up -d --build
```

## Best Practices Implemented

[x] There are unit tests (1 point)
[ ] There is an integration test (1 point)
[x] Linter and/or code formatter are used (1 point)
[x] There's a Makefile (1 point)
[x] There are pre-commit hooks (1 point)
[ ] There's a CI/CD pipeline (2 points)

### Unit test

- tests/test_model.py

### Pylint

- added ```pyproject.toml``` file to suppress inutile Pylint messages

- now run checks (should get 10.00/10):

```bash
pylint --recursive=y scripts/
```
### Makefile

- makefile ```Makefile``` in the main dir
