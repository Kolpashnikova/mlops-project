import pandas as pd
import pickle
import click
from prefect import task, flow
import json

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DatasetCorrelationsMetric, ColumnQuantileMetric, ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, RegressionQualityMetric

from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, CounterAgg, PanelValue, PlotType, ReportFilter, DashboardPanelTestSuite, TestSuitePanelType
from evidently.renderers.html_widgets import WidgetSize
from evidently.test_suite import TestSuite
from evidently.test_preset import RegressionTestPreset
from evidently.ui.workspace.cloud import CloudWorkspace


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import os

EVIDENTLY_API_KEY = os.getenv('EVIDENTLY_API_KEY')
EVIDENTLY_TEAM_ID = os.getenv('EVIDENTLY_TEAM_ID')

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

# @flow
def run_report(model_file: str):

    df_train_df = pd.read_parquet('data/atus37.parquet')
    
    X_train, X_val, y_train, y_val = prepare_data(df_train_df)

    df_train, df_val, num_features, cat_features = preprocess(X_train, X_val)

    model = load_pickle(model_file)

    X_train['prediction'] = model.predict(df_train)
    X_val['prediction'] = model.predict(df_val)

    X_train['target'] = y_train
    X_val['target'] = y_val

    column_mapping = ColumnMapping(
        target='target',
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features
    )

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DatasetCorrelationsMetric(), 
        ColumnQuantileMetric(column_name="age", quantile = 0.5),
        RegressionQualityMetric(),
    ]
    )

    report.run(reference_data=X_train, current_data=X_val, column_mapping=column_mapping)


    regression_performance = TestSuite(tests=[
        RegressionTestPreset(),
    ])
    
    regression_performance.run(reference_data=X_train, current_data=X_val)

    # result = report.as_dict()

    return report, regression_performance

# @click.command()
# @click.option(
#     "--model_file",
#     default="model/model.pkl",
#     help="Model filename"
# )
# flow
def run_batch_monitoring(model_file="model/model.pkl"):
    report, regression_performance = run_report(model_file)

    # # Specify the file name
    # file_name = "output/dictionary.json"

    # # Write dictionary to a file
    # with open(file_name, 'w') as file:
    #     json.dump(report, file, indent=4)

    ws = CloudWorkspace(token=EVIDENTLY_API_KEY, url="https://app.evidently.cloud")

    project = ws.create_project('Wages model monitoring', team_id=EVIDENTLY_TEAM_ID)
    project.description = 'This project is used to monitor the model for predicting wages'
    project.save()

    ws.add_report(project.id, report)
    ws.add_report(project.id, regression_performance)


    # drift_score = result['metrics'][0]['result']['drift_score']
    # dataset_drift = result['metrics'][1]['result']['share_of_drifted_columns']
    # missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    # correlation = result['metrics'][3]['result']['current']['stats']['pearson']['abs_max_features_correlation']
    # quantile = result['metrics'][4]['result']['current']['value']
    # current_rsme = result['metrics'][5]['result']['current']['rmse']
    # reference_rsme = result['metrics'][5]['result']['reference']['rmse']

    #configure the dashboard
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Wages Model Monitoring Dashboard",
        )
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            title="Drift Score",
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    field_path="drift_score",
                    legend="score"
                ),
            ],
            plot_type=PlotType.BAR,
            size=WidgetSize.HALF,
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            title="Dataset Drift",
            values=[
                PanelValue(
                    metric_id="DatasetDriftMetric",
                    field_path="share_of_drifted_columns",
                    legend="share"
                ),
            ],
            plot_type=PlotType.BAR,
            size=WidgetSize.HALF,
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            title="Number of Missing Values",
            values=[
                PanelValue(
                    metric_id="DatasetMissingValuesMetric",
                    field_path="current.share_of_missing_values",
                    legend="share"
                ),
            ],
            plot_type=PlotType.LINE,
            size=WidgetSize.HALF,
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            title="Regression RMSE (Current)",
            values=[
                PanelValue(
                    metric_id="RegressionQualityMetric",
                    field_path="current.rmse",
                    legend="rmse"
                ),
            ],
            plot_type=PlotType.BAR,
            size=WidgetSize.HALF,
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            title="Regression RMSE (Reference)",
            values=[
                PanelValue(
                    metric_id="RegressionQualityMetric",
                    field_path="reference.rmse",
                    legend="rmse"
                ),
            ],
            plot_type=PlotType.BAR,
            size=WidgetSize.HALF,
        ),
    )

    project.save()

    return 

if __name__ == '__main__':
    run_batch_monitoring()