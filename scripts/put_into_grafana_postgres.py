import datetime
import time
import random
import logging 
import pandas as pd
import psycopg
import pickle

from prefect import task, flow
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, DatasetCorrelationsMetric, ColumnQuantileMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	current_data varchar(255),
	drift_score float,
	dataset_drift float,
	missing_values float,
	correlation float,
	quantile float
)
"""

begin = 2003

# @task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
# @task
def prepare_data(data):
    target_column = 'earnweek'  
    data = data.dropna(subset=[target_column])
    X = data.drop(columns = target_column)  
    y = data[target_column]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# @task
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

    return X_train, X_val, numerical_features, categorical_features

# @task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='mlops_grafana'")
		if len(res.fetchall()) == 0:
			conn.execute("create database mlops_grafana;")
		with psycopg.connect("host=localhost port=5432 dbname=mlops_grafana user=postgres password=example") as conn:
			conn.execute(create_table_statement)

# @task
def calculate_metrics_postgresql(curr, i, model_file="model/model.pkl"):	

    df_train_df = pd.read_parquet('data/atus37.parquet')
    
    X_train, X_val, y_train, y_val = prepare_data(df_train_df)
	
    if i == "train":
        current_data = X_train
        reference_data = X_val
    else:
        current_data = X_val
        reference_data = X_train

    df_train, df_val, num_features, cat_features = preprocess(X_train, X_val)

    model = load_pickle(model_file)
	
    if i == "train":
        current_data['prediction'] = model.predict(df_train)
        reference_data['prediction'] = model.predict(df_val)
    else:
        current_data['prediction'] = model.predict(df_val)	
        reference_data['prediction'] = model.predict(df_train)	

    column_mapping = ColumnMapping(
        target=None,
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features
    )

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DatasetCorrelationsMetric(), 
        ColumnQuantileMetric(column_name="age", quantile = 0.5)
    ]
    )

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    result = report.as_dict()

    drift_score = result['metrics'][0]['result']['drift_score']
    dataset_drift = result['metrics'][1]['result']['share_of_drifted_columns']
    missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    correlation = result['metrics'][3]['result']['current']['stats']['pearson']['abs_max_features_correlation']
    quantile = result['metrics'][4]['result']['current']['value']

    curr.execute(
        "insert into dummy_metrics(current_data, drift_score, dataset_drift, missing_values, correlation, quantile) values (%s, %s, %s, %s, %s, %s)",
        (i, drift_score, dataset_drift, missing_values, correlation, quantile)
    )

# @flow
def run_batch_monitoring():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=mlops_grafana user=postgres password=example", autocommit=True) as conn:
		for i in ["train", "val"]:			
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	run_batch_monitoring()