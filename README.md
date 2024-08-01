# MLOps Project





### Run Dockerized Model

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

### Best Practices Implemented

[x] There are unit tests (1 point)
[ ] There is an integration test (1 point)
[x] Linter and/or code formatter are used (1 point)
[x] There's a Makefile (1 point)
[x] There are pre-commit hooks (1 point)
[x] There's a CI/CD pipeline (2 points)

### Unit test

- tests/test_model.py

### Pylint 

- added ```pyproject.toml``` file to suppress inutile Pylint messages

- now run checks (should get 10.00/10):

```bash
pylint --recursive=y scripts/
```





