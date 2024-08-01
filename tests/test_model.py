import scripts.predict_local as predict
import pandas as pd

def test_predict():

    features = {
        'year': 2010, 
        'age': 46, 
        'sex': "Male", 
        'race': "Asian only", 
        'hispan': "Not Hispanic", 
        'yrimmig': "Not foreign born", 
        'citizen': "Native, born in United States", 
        'bpl': "U.S., n.s.",
        'mbpl': "China", 
        'fbpl': "China", 
        'educ': "Master's degree (MA, MS, MEng, MEd, MSW, etc.)", 
        'educyrs': "Master's degree--three+ year program", 
        'empstat': "Employed - at work", 
        'multjobs': "No", 
        'clwkr': "Private, for profit",
        'occ2': "Computer and mathematical science occupations", 
        'occ': 1021, 
        'ind2': "Professional, scientific, and technical services", 
        'ind': 7380, 
        'whyabsnt': "NIU (Not in universe)", 
        'fambus_pay': "NIU (Not in universe)", 
        'fambus_wrk': "NIU (Not in universe)",
        'looking': "NIU (Not in universe)", 
        'retired': "NIU (Not in universe)", 
        'fullpart': "Full time",
        'hourwage': 14
    }

    features = pd.DataFrame([features])

    prediction = predict.predict(features)

    assert prediction == 1904.9652488273023
