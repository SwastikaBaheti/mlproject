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
                "KNN Regressor": KNeighborsRegressor(),
                "RandomForest Regressor": RandomForestRegressor(),
                "Adaboost Regressor": AdaBoostRegressor(),
                "GradientBoost Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor()
            }

            logging.info('Evaluating the models')
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info('Model evaluation completed')

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info('Found the model with maximum accuracy')

            logging.info('Saving the regression model')
            save_model(file_path=self.model_trainer_config.trained_model_file_path, model=best_model)

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
 
            return(score)
        except Exception as e:
            raise CustomException(e, sys)
