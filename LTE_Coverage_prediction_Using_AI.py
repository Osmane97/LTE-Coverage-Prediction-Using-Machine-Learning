
import numpy as np
import streamlit as st
import pandas as pd
from joblib import load


#Load best model
best_models = {'RSRP': load("C:/Users/obouk/OneDrive/Bureau/Best_model_for_RSRP.joblib"),
               'SINR': load("C:/Users/obouk/OneDrive/Bureau/Best_model_for_SINR.joblib"),
               'RSRQ': load("C:/Users/obouk/OneDrive/Bureau/Best_model_for_RSRQ.joblib")
               }

#Define error range from Test_RMSE model results



Signal_Strength_threshold = {
    'RSRP': {
        'Excellent': (-float('inf'), -80),
        'Good': (-81, -90),
        'Fair': (-91, -100),
        'Poor': (-101, -float('inf'))
    },
    'SINR': {
        'Excellent': (20, float('inf')),
        'Good': (13, 19),
        'Fair': (0, 12),
        'Poor': (-float('inf'), -1)
    },
    'RSRQ': {
        'Excellent': (float('inf'), -5),
        'Good': (-9, -4),
        'Fair': (-12, -8),
        'Poor': (-float('inf'), -13)
    }
}

# Define color mapping
quality_colors = {
    'Excellent': 'background-color: green; color: white',
    'Good': 'background-color: yellow',
    'Fair': 'background-color: orange; color: white',
    'Poor': 'background-color: red; color: white'
}


#Input features
input_features = ['Distance (miles)', 'HBA', 'Azimuth']


RMSE_model_df = pd.DataFrame( {
    'Metrics':['RSRP','SINR','RSRQ'],
    '± Error': [3.4,2.34, 0.81]
})
#Core Function Development


def get_prediction(input_features):
    # below condition is built to reshape our input in 2D array
    # in case we will predict only one row instead of batch
    global quality
    input_features =np.array(input_features)
    if len(input_features.shape) == 1:
        input_features = input_features.reshape(1, -1)

    result = {}  # store results

    for target in best_models.keys():
        model = best_models[target]  # select best model for specific target
        pred = model.predict(input_features)[0]  # predict from input attributes by extracting scalar value
        for qual, (low, high) in Signal_Strength_threshold[target].items():
            if low <= pred <= high:
                quality = qual
                break

        result[target] = {
            'Prediction': round(pred,2),
            'Quality': quality
        }
    return result


#User interface with streamlit

st.title('LTE Signal Strength Prediction')

with st.sidebar:
    st.subheader('Model error accuracy')
    st.table(RMSE_model_df)
    st.caption('These represents prediction error ranges for each metric')

st.subheader('Select Prediction Mode')
#Create two buttons one for single prediction and the other for batch prediction
prediction_mode = st.radio('', ['Single Prediction', 'Batch Prediction'], horizontal=True)

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
            'Metrics': ['RSRP', 'SINR', 'RSRQ'],
            'Value': [
                f"{results['RSRP']['Prediction']} ",
                f"{results['SINR']['Prediction']} ",
                f"{results['RSRQ']['Prediction']} "
            ],
            'Quality': [results['RSRP']['Quality'],
                        results['SINR']['Quality'],
                        results['RSRQ']['Quality']]
        }
        prediction_df = pd.DataFrame(targets_results)

        st.markdown('Prediction Results')
        st.table(prediction_df)

elif prediction_mode == 'Batch Prediction':
    st.subheader('Upload your CSV file')
    uploaded_file = st.file_uploader('')
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
                'RSRP Prediction': f"{results['RSRP']['Prediction']}",
                'RSRP Quality':results['RSRP']['Quality'],
                'SINR Prediction': f"{results['SINR']['Prediction']}",
                'SINR Quality':results['SINR']['Quality'],
                'RSRQ Prediction':f"{results['RSRQ']['Prediction']}",
                'RSRQ Quality':results['RSRQ']['Quality']
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

