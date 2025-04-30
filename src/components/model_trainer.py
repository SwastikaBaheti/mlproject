import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_model

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data, test_data):
        logging.info('Entered the model trainer component')
        try:
            logging.info('Splitting the train and test data')
            X_train = train_data[:,:-1]
            y_train = train_data[:,-1]
            X_test = test_data[:,:-1]
            y_test = test_data[:,-1]

            logging.info('Creating the model dictionary')
            models = {
                "Linear Regressor": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "RandomForest Regressor": RandomForestRegressor(),
                "Adaboost Regressor": AdaBoostRegressor(),
                "GradientBoost Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor()
            }

            params={
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2'],
                },
                "RandomForest Regressor":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regressor":{},
                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Adaboost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            logging.info('Evaluating the models')
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            logging.info('Model evaluation completed')

            best_model_name = max(model_report, key=lambda k: model_report[k][0])
            best_model_score, best_model_params = model_report[best_model_name]

            best_model = models[best_model_name].__class__(**best_model_params)
            
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info('Found the model with maximum accuracy')

            logging.info('Saving the regression model')
            save_model(file_path=self.model_trainer_config.trained_model_file_path, model=best_model)

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
 
            return(score)
        except Exception as e:
            raise CustomException(e, sys)
