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