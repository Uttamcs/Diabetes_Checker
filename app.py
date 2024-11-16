import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("diabetes.csv")

st.title("Diabetes CheckUp")
st.subheader("Training Data")
st.write(data.describe())

st.subheader("Visualization")
st.bar_chart(data)

x = data.drop(["Outcome"], axis=1)
y = data["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def user_report():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0, 67, 20)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.4, 0.47)
    age = st.sidebar.slider("Age", 21, 88, 33)

    user_report_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_report()

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

st.subheader("Accuracy:")
st.write(str(accuracy_score(y_test, rf.predict(x_test)) * 100) + "%")

user_result = rf.predict(user_data)

st.subheader("Your Report:")
output = "You are not Diabetic" if user_result[0] == 0 else "You are Diabetic"
st.write(output)
