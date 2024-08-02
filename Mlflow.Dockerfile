FROM ghcr.io/mlflow/mlflow:v2.3.1

COPY [ "mlruns", "./" ]

COPY [ "mlartifacts", "./" ]

COPY [ "mlflow.db", "./" ]

EXPOSE 5000

ENTRYPOINT ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "0.0.0.0"]
