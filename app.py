from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


app = Flask(__name__)

# the best model
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    designation_encoding = {"Analyst": 1, "Senior Analyst": 2, "Associate": 3, "Manager": 4, "Senior Manager": 5, "Director": 6}
    sex_encoding = {"F": 0, "M": 1}
    unit_encoding = {"Marketing": 1, "Finance": 2, "Web": 3, "Management": 4, "Operations": 5, "IT": 6}

    designation = designation_encoding[data['designation']]
    sex = sex_encoding[data['sex']]
    unit = unit_encoding[data['unit']]

    doj = datetime.strptime(data['doj'], '%Y-%m-%d')
    current_date = datetime.strptime(data['current_date'], '%Y-%m-%d')
    days_at_company = (current_date - doj).days
    overall_experience = days_at_company +  (data['past_exp'] * 365)

    features = pd.DataFrame([{
        'sex' : sex,
        'designation' : designation,
        'age' : data['age'],
        'unit' : unit,
        'leaves used' : data['leaves_used'],
        'leaves remaining' : data['leaves_remaining'],
        'ratings' : data['ratings'],
        'overall_experience' : overall_experience
    }])
    
    prediction = model.predict(features)

    return jsonify(prediction = np.round(prediction[0],2))

# commented out for deploying in render cloud services
# if __name__ == '__main__':
#     app.run(debug=True)
