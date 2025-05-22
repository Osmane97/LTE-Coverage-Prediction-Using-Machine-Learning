import warnings
import numpy as np
import streamlit as st
import pandas as pd
from joblib import load

#Suppress sklearn warnings about feature names


#Load best model
best_models = {'RSRP': load("C:/Users/obouk/OneDrive/Bureau/Best_model_for_RSRP.joblib"),
               'SINR': load("C:/Users/obouk/OneDrive/Bureau/Best_model_for_SINR.joblib"),
               'RSRQ': load("C:/Users/obouk/OneDrive/Bureau/Best_model_for_RSRQ.joblib")
               }

#Define error range from Test_RMSE model results



Signal_Strength_threshold = pd.DataFrame( {
    'Quality' : ['Excellent', 'Good', 'Fair', 'Poor'],
    'RSRP (dBm)': ['≥ -80', '-90 to -80', '-100 to -90', '≤ -100'],
    'SINR (dB)': ['≥ 20', '13 to 20', '0 to 13', '≤ 0'],
    'RSRQ (dB)': [ '≥ -10', '-15 to -10', '-20 to -15', '≤ -20']
}). set_index ('Quality')


#Input features
input_features = ['Distance (miles)', 'HBA', 'Azimuth']


RMSE_model_df = pd.DataFrame( {
    'Metrics':['RSRP','SINR','RSRQ'],
    '± Error': [3.4,2.34, 0.81]
})
#Core Function Development


def get_prediction(input):
    # below condition is built to reshape our input in 2D array
    # in case we will predict only one row instead of batch
    input =np.array(input)
    if len(input.shape) == 1:
        input = input.reshape(1, -1)

    result = {}  # store results

    for target in best_models.keys():
        model = best_models[target]  # select best model for specific target
        pred = model.predict(input)[0]  # predict from input attributes by extracting scalar value
        result[target] = {
            'Prediction': round(pred,2),
        }
    return result


#User interface with streamlit

st.title('LTE Signal Strength Prediction')

with st.sidebar:
    st.subheader('Model error accuracy')
    st.table(RMSE_model_df)
    st.caption('These represents prediction error ranges for each metric')

    st.subheader('Signal Strength Thresholds')
    st.table(Signal_Strength_threshold)


st.subheader('Select Prediction Mode')
#Create two buttons one for single prediction and the other for batch prediction
prediction_mode = st.radio('Prediction Mode', ['Single Prediction', 'Batch Prediction'], horizontal=True)

if prediction_mode == 'Single Prediction':
    st.subheader('Enter below features')

    distance = st.number_input('Distance (miles)', min_value=0.0, max_value=18.6, format='%.2f', step=0.1)
    hba = st.number_input('Tower Height (meters)', min_value=4.0, max_value=72.0, format='%.1f', step=0.1)
    azimuth = st.number_input('Azimuth (degress)', min_value=0, max_value=360, step=5)

    submit = st.button('Predict')
    if submit:
        input_features = [[distance, hba, azimuth]]
        results = get_prediction(input_features)


        #Preparing result to display them in a table
        targets_results = {
            'Metrics': ['RSRP (dBm)', 'SINR (dB)', 'RSRQ (dB)'],
            'Value': [
                f"{results['RSRP']['Prediction']}",
                f"{results['SINR']['Prediction']}",
                f"{results['RSRQ']['Prediction']}"
            ],
        }
        prediction_df = pd.DataFrame(targets_results)

        st.markdown('Prediction Results')
        st.dataframe(prediction_df)

elif prediction_mode == 'Batch Prediction':
    st.subheader('Upload your CSV file')
    uploaded_file = st.file_uploader('Upload CSV file')
    submit = st.button('Predict')

    if submit and uploaded_file:
        CSV_df = pd.read_csv(uploaded_file)
        #predict each row within the csv file
        predictions_results = []

        for _, row in CSV_df.iterrows():
            input_csv_data = [[row['Distance (miles)'],
                               row['HBA'],
                               row[ 'Azimuth']
                               ]
                              ]
            results = get_prediction(input_csv_data)

            results_by_row = {
                'UE_longitude': row.get('UE_longitude'),
                'User_latitude': row.get('User_latitude'),
                'eNodeB_ID': int(row.get('eNodeB_ID')),
                'Distance (miles)': f"{ row['Distance (miles)']:.2f}",
                'HBA': f"{row['HBA']:.1f}",
                'Azimuth': int(row[ 'Azimuth']),
                'RSRP Prediction (dBm)': f"{results['RSRP']['Prediction']}",
                'SINR Prediction (dB)': f"{results['SINR']['Prediction']}",
                'RSRQ Prediction (dB)':f"{results['RSRQ']['Prediction']}",
            }
            predictions_results.append(results_by_row)

        prediction_df = pd.DataFrame(predictions_results)

        st.markdown('Prediction Results')
        st.table(prediction_df)
        st.download_button('Download Prediction Results',
                           data = prediction_df.to_csv(index= False),
                           mime = 'csv',
                           file_name= 'LTE_Signal_Strength_Prediction_Results')

    else:
        st.warning('Please Upload your dataset')

