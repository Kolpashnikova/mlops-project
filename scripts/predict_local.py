import pandas as pd
import pickle
# import os
# import boto3
from flask import Flask, request, jsonify

# SPACES_KEY = os.getenv('SPACES_KEY')
# SPACES_SECRET = os.getenv('SPACES_SECRET')

# session = boto3.session.Session()
# client_spaces = session.client('s3',
#                         region_name='nyc3',
#                         endpoint_url='https://nyc3.digitaloceanspaces.com',
#                         aws_access_key_id=os.getenv('SPACES_KEY'),
#                         aws_secret_access_key=os.getenv('SPACES_SECRET'))

def run_download_preprocessor():
    # response_1 = client_spaces.get_object(Bucket='mlops-project', Key='output/preprocessor.pkl')

    # with open("output/preprocessor.pkl", 'wb') as file:
    #     file.write(response_1['Body'].read())

    with open('output/preprocessor.pkl', 'rb') as f_in:
        preprocessor = pickle.load(f_in)

    return preprocessor

def run_download_model():

    # response_2 = client_spaces.get_object(Bucket='mlops-project', Key='model/model.pkl')

    # with open("model/model.pkl", 'wb') as file:
    #     file.write(response_2['Body'].read())
    
    with open('model/model.pkl', 'rb') as f_in:
        model = pickle.load(f_in)

    return model


def predict(features):
    preprocessor = run_download_preprocessor()
    X = preprocessor.transform(features)
    model = run_download_model()
    preds = model.predict(X)
    return float(preds[0])

app = Flask('wage-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    features = request.get_json()

    features = pd.DataFrame([features])
    
    pred = predict(features)

    result = {
        'wage': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)