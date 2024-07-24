import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('E:/Python Projects/HR Prediction/HR_comma_sep.csv')
    df = df.drop('dept', axis=1)
    label_encoder = LabelEncoder()
    df['salary'] = label_encoder.fit_transform(df['salary'])
    df = pd.get_dummies(df, columns=['salary'], prefix='salary')
    X = df.drop('left', axis=1)
    y = df['left']
    scaler = StandardScaler()
    numerical_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return X, y

# Load the trained model
@st.cache_data
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def main():
    st.title("HR Attrition Prediction")

    X, y = load_data()

    # Get user input
    satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    last_evaluation = st.number_input("Last Evaluation Score", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    number_project = st.number_input("Number of Projects", min_value=0, value=3, step=1)
    average_montly_hours = st.number_input("Average Monthly Hours", min_value=0, value=160, step=1)
    time_spend_company = st.number_input("Time Spent in Company (Years)", min_value=0, value=2, step=1)
    Work_accident = st.number_input("Work Accident (0 or 1)", min_value=0, max_value=1, value=0, step=1)
    promotion_last_5years = st.number_input("Promotion in Last 5 Years (0 or 1)", min_value=0, max_value=1, value=0, step=1)
    salary_level = st.radio("Salary Level", options=["Low", "Medium", "High"], index=0)

    # Convert salary level to one-hot encoding
    salary_low, salary_medium, salary_high = 0, 0, 0
    if salary_level == "Low":
        salary_low = 1
    elif salary_level == "Medium":
        salary_medium = 1
    else:
        salary_high = 1

    input_data = [[satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary_low, salary_medium, salary_high]]

    if st.button("Predict"):
        loaded_model = load_model()
        prediction = loaded_model.predict(input_data)
        if prediction[0] == 0:
            st.success("The model predicts that the employee is not likely to leave the company.")
        else:
            st.warning("The model predicts that the employee is likely to leave the company.")

if __name__ == "__main__":
    main()