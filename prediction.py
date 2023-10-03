#!/usr/bin/env python
# coding: utf-8


import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def encoder(input_val, column):
    label_encoder = joblib.load(r'Encoder/label_encoder.joblib')
    ordinal_encoder = joblib.load(r'Encoder/ordinal_encoder.joblib')
    if column in label_encoder:
        encoder = label_encoder[column]
        value = encoder.transform([input_val])
        return value[0]
    else:
        encoder = ordinal_encoder[column]
        value = encoder.transform([[input_val]])
        return value[0][0]
    

def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)


