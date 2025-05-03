from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for a home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for making predictions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        input_data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score')
        )
        input_data_df = input_data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        predicted_value = predict_pipeline.predict(input_data_df)

        return render_template('form.html', results=predicted_value[0])
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')