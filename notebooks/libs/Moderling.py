import pandas as pd
import numpy as np
import itertools
import sklearn.metrics as met
import xgboost as xgb

def bagging(vote_number, prediction_list):
    total_prediction = []
    for i in range(len(prediction_list[0])):
        voters = 0
        
        for prediction in prediction_list:
            if prediction[i] == 1:
                voters += 1
                
        if voters >= vote_number:
            total_prediction.append(1)
        else:
            total_prediction.append(0)
                
    return np.array(total_prediction
