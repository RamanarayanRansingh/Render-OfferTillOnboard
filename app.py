from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model, scaler, and columns
model = joblib.load('models/xgb_tunedv2.pkl')
scaler = joblib.load('scalers/scaler.pkl')
columns = joblib.load('models/columns.pkl')

def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical features
    categorical_columns = ['DOJ_Extended', 'Offered_Band', 'Joining_Bonus', 'Candidate_relocate_actual', 
                           'Gender', 'Candidate_Source', 'LOB', 'Location', 'Region_Name', 'Domicile_Name']
    input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
    
    # Ensure all columns present during training are in the input data
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Scale numeric features
    numeric_features = ['Duration_to_accept_offer', 'Notice_Period', 'Percent_hike_expected_in_CTC', 
                        'Percent_difference_CTC', 'Rex_in_Yrs', 'Age']
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    
    input_df = input_df[columns]  # Reorder columns to match the model training
    
    return input_df

def make_prediction(input_data):
    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)
    return "Will Join" if prediction[0] == 1 else "May Not Join"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        result = make_prediction(input_data)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)