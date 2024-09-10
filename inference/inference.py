import gradio as gr
import pandas as pd
import joblib

# Load the model, scaler, and columns
model = joblib.load('models/xgb_tunedv3.pkl')
scaler = joblib.load('models/scaler.pkl')
columns = joblib.load('models/columns.pkl')

# Preprocessing function
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

# Prediction function
def make_prediction(input_data):
    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)
    return "Yes" if prediction[0] == 0 else "No"

# Gradio interface function
def inference_page():
    with gr.Blocks() as demo:
        # Define inputs
        DOJ_Extended = gr.Dropdown(choices=["Yes", "No"], label="DOJ Extended")
        Duration_to_accept_offer = gr.Slider(minimum=0, maximum=224, label="Duration to Accept Offer (days)")
        Notice_Period = gr.Slider(minimum=0, maximum=120, label="Notice Period (days)")
        Offered_Band = gr.Dropdown(choices=["E1", "E2", "E3", "E4"], label="Offered Band")
        Percent_hike_expected_in_CTC = gr.Slider(minimum=-68, maximum=359, label="Percent Hike Expected in CTC")
        Percent_difference_CTC = gr.Slider(minimum=-67, maximum=300, label="Percent Difference in CTC")
        Joining_Bonus = gr.Dropdown(choices=["Yes", "No"], label="Joining Bonus")
        Candidate_relocate_actual = gr.Dropdown(choices=["Yes", "No"], label="Candidate Relocate")
        Gender = gr.Dropdown(choices=["Male", "Female"], label="Gender")
        Candidate_Source = gr.Dropdown(choices=["Direct", "Employee Referral", "Agency"], label="Candidate Source")
        LOB = gr.Dropdown(choices=["ERS", "INFRA", "Healthcare", "BFSI", "CSMP", "ETS", "AXON", "EAS", "MMS"], label="Line of Business (LOB)")
        Location = gr.Dropdown(choices=["Noida", "Chennai", "Mumbai", "Hyderabad", "Kolkata", "Gurgaon", "Bangalore", "Cochin", "Pune", "Ahmedabad"], label="Location")
        Region_Name = gr.Dropdown(choices=["North", "South", "East", "West"], label="Region Name")
        Domicile_Name = gr.Dropdown(choices=["Andhra Pradesh", "Bihar", "Chandigarh", "Chhattisgarh", "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Odisha", "Punjab", "Rajasthan", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarkhand", "West Bengal"], label="Domicile Name")
        Age = gr.Slider(minimum=20, maximum=60, label="Age")
        
        # Prediction button and result output
        predict_button = gr.Button("Predict")
        result = gr.Textbox(label="Prediction Result")

        # Collect inputs into a dictionary
        def collect_and_predict(DOJ_Extended, Duration_to_accept_offer, Notice_Period, Offered_Band,
                                Percent_hike_expected_in_CTC, Percent_difference_CTC, Joining_Bonus, 
                                Candidate_relocate_actual, Gender, Candidate_Source, LOB, Location, 
                                Region_Name, Domicile_Name, Age):
            input_data = {
                'DOJ_Extended': DOJ_Extended,
                'Duration_to_accept_offer': Duration_to_accept_offer,
                'Notice_Period': Notice_Period,
                'Offered_Band': Offered_Band,
                'Percent_hike_expected_in_CTC': Percent_hike_expected_in_CTC,
                'Percent_difference_CTC': Percent_difference_CTC,
                'Joining_Bonus': Joining_Bonus,
                'Candidate_relocate_actual': Candidate_relocate_actual,
                'Gender': Gender,
                'Candidate_Source': Candidate_Source,
                'LOB': LOB,
                'Location': Location,
                'Region_Name': Region_Name,
                'Domicile_Name': Domicile_Name,
                'Age': Age
            }
            return make_prediction(input_data)

        # Button click action
        predict_button.click(collect_and_predict, 
                             inputs=[DOJ_Extended, Duration_to_accept_offer, Notice_Period, Offered_Band, 
                                     Percent_hike_expected_in_CTC, Percent_difference_CTC, Joining_Bonus, 
                                     Candidate_relocate_actual, Gender, Candidate_Source, LOB, Location, 
                                     Region_Name, Domicile_Name, Age], 
                             outputs=result)

    return demo

# Run the Gradio app
if __name__ == "__main__":
    inference_page().launch()
