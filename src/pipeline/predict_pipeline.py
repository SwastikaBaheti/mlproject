import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import get_preprocessor_object, get_model_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, input_features):
        try:
            preprocessor_obj = get_preprocessor_object(file_path='artifacts\preprocessor.pkl')
            model_obj = get_model_object(file_path='artifacts\model.pkl')

            input_data_scaled = preprocessor_obj.transform(input_features)
            print(input_data_scaled)
            prediction = model_obj.predict(input_data_scaled)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                gender: str, 
                race_ethnicity: int, 
                parental_level_of_education: str, 
                lunch: str, 
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)