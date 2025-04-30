import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_preprocessor_object
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_featues = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            logging.info('Creating numerical and categorical pipelines')
            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='median')),
                    ("StandardScaler", StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoder", OneHotEncoder())
                ]
            )

            logging.info('Setting up Column Transformer by combining numerical and categorical pipelines')
            col_transformer = ColumnTransformer([
                ("NumericalPipeline", numerical_pipeline, numerical_featues),
                ("CategoricalPipeline", categorical_pipeline, categorical_features)
            ])

            return col_transformer
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_data_path, test_data_path):
        logging.info('Entered the data transformation component')
        try:
            logging.info('Initiated train and test data reading')
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info('Train and test data read sucessfully')

            logging.info('Getting the Column Transformer object')
            col_transformer = self.get_data_transformer()

            target_column = 'math_score'

            input_feature_train_df = train_data.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_data[target_column]

            input_feature_test_df = test_data.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_data[target_column]

            logging.info('Applying Column Transformer on the train dataset')
            input_train_data_transformed = col_transformer.fit_transform(input_feature_train_df)

            logging.info('Applying Column Transformer on the test dataset')
            input_test_data_transformed = col_transformer.transform(input_feature_test_df)

            train_arr = np.c_[input_train_data_transformed, np.array(target_feature_train_df)]
            test_arr = np.c_[input_test_data_transformed, np.array(target_feature_test_df)]
            logging.info('Preprocessing completed')

            logging.info('Saving the Column Transformer object into a pickle file')
            save_preprocessor_object(file_path=self.transformation_config.preprocessor_obj_file_path, preprocessor_obj=
                                          col_transformer)
            
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)