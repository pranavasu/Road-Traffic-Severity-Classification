#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import encoder, get_prediction


model = joblib.load(r'Model/random_forest_model.pkl')

st.set_page_config(page_title="Accident Severity Prediction App", page_icon="🚧", layout="wide")

# creating an option list for the drop down menus

options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_time_cat = ['evening', 'night', 'afternoon', 'morning']
options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
# options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
#        'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
#        'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
#        'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_vehicle_type = ['Automobile','Public (> 45 seats)','Lorry (41?100Q)','Public (13?45 seats)','Lorry (11?40Q)',
       'Long lorry','Public (12 seats)','Taxi','Pick up upto 10Q','Stationwagen','Ridden horse','Other',
       'Bajaj', 'Turbo','Motorcycle', 'Special vehicle','Bicycle']
options_junction_type = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other','Unknown', 'T Shape', 'X Shape']
options_no_vehicles = [2, 1, 3, 6, 4, 7]
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_casualities = [2, 1, 3, 4, 6, 5, 8, 7]


features = ['Area_accident_occured', 'Time_Category', 'Driving_experience','Age_band_of_driver', 'Type_of_vehicle', 'Types_of_Junction',
       'Number_of_vehicles_involved', 'Cause_of_accident', 'Day_of_week','Number_of_casualties']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):
        st.subheader("Enter the input for following features:")

        Area_accident_occured = st.selectbox("Accident Area", options=options_acc_area)
        Time_Category = st.selectbox("Time category of the day", options=options_time_cat)
        Driving_experience = st.selectbox("Driving Experience: ", options=options_driver_exp)
        Age_band_of_driver = st.selectbox("Driver Age: ", options=options_age)
        Type_of_vehicle = st.selectbox("Vehicle Type: ", options=options_vehicle_type)
        Types_of_Junction = st.selectbox("Junction Type: ", options=options_junction_type)
        Number_of_vehicles_involved = st.slider("Vehicles Involved: ", 1, 7, value=1, format="%d")
        Cause_of_accident = st.selectbox("Cause of Accident: ", options=options_cause)
        Day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        Number_of_casualties = st.slider("No of Casualties: ", 1,8, value=1, format="%d")
    
        submit = st.form_submit_button("Predict")

    if submit:
        accident_area = encoder(Area_accident_occured, 'Area_accident_occured')
        time_cat = encoder(Time_Category, 'Time_Category')
        driving_experience = encoder(Driving_experience, 'Driving_experience')
        driver_age = encoder(Age_band_of_driver, 'Age_band_of_driver')
        vehicle_type = encoder(Type_of_vehicle, 'Type_of_vehicle')
        junction_type = encoder(Types_of_Junction, 'Types_of_Junction')
        number_of_vehicles_involved = encoder(Number_of_vehicles_involved, 'Number_of_vehicles_involved')
        accident_cause = encoder(Cause_of_accident, 'Cause_of_accident')
        day_of_week = encoder(Day_of_week, 'Day_of_week')
        number_of_casualties = encoder(Number_of_casualties, 'Number_of_casualties')
        
        data = np.array([accident_area,time_cat,driving_experience,driver_age,vehicle_type, junction_type,number_of_vehicles_involved,accident_cause,day_of_week,number_of_casualties]).reshape(1,-1)
        pred = get_prediction(data=data, model=model)
        if pred == 0:
              st.write(f"The severity is a Fatal Injury")
        elif pred == 1:
              st.write(f"The severity is a Serious Injury")
        else:
              st.write(f"The severity is a Slight Injury")
       #st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()





