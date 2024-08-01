FROM python:3.9-slim

RUN pip install -U pip

WORKDIR /app

COPY [ "requirements_freeze.txt", "./" ]

RUN pip install -r requirements_freeze.txt

COPY [ "scripts/predict.py", "./" ]

COPY [ "model/model.pkl", "./" ]

COPY [ "output/preprocessor.pkl", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

