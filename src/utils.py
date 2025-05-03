import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

def save_preprocessor_object(file_path, preprocessor_obj):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'wb') as fp:
                 dill.dump(preprocessor_obj, fp)

        except Exception as e:
            raise CustomException(e, sys)
        
def get_preprocessor_object(file_path):
        try:
            with open(file_path, 'rb') as fp:
                processor_obj = dill.load(fp)
            return processor_obj
        except Exception as e:
            raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
     try:
        model_report = dict()
        for i in range(0, len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model_param = params[model_name]

            # Applying Hyperparameter tunning
            random_cv = RandomizedSearchCV(estimator=model, param_distributions=model_param, cv=3, n_jobs=-1)

            # Model Training
            random_cv.fit(X_train, y_train)

            # Model Prediction
            y_pred_test = random_cv.predict(X_test)

            # Model Evaluation
            score_test = get_model_accuracy(y_test, y_pred_test)

            # Storing the results
            model_report[model_name] = (score_test, random_cv.best_params_)

        return model_report
     except Exception as e:
        raise CustomException(e, sys)
     
def get_model_accuracy(y_true, y_pred):
    try:
        return r2_score(y_true, y_pred)    
    except Exception as e:
        raise CustomException(e, sys)
    
def save_model(file_path, model):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'wb') as fp:
                 dill.dump(model, fp)

        except Exception as e:
            raise CustomException(e, sys)
        
def get_model_object(file_path):
        try:
            with open(file_path, 'rb') as fp:
                model_obj = dill.load(fp)
            return model_obj
        except Exception as e:
            raise CustomException(e, sys)