import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill

def save_preprocessor_object(file_path, preprocessor_obj):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'wb') as fp:
                 dill.dump(preprocessor_obj, fp)

        except Exception as e:
            raise CustomException(e, sys)