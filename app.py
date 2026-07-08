import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title(" Titanic Survival Dashboard")
st.write("Explore the data and predict survival for a hypothetical passenger.")

df = pd.read_csv("data/train.csv")
model = joblib.load("model.pkl")

tab1, tab2 = st.tabs([" Data Explorer", " Predict Survival"])

with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(20))

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/survival_by_class.png")
    with col2:
        st.image("images/survival_by_sex.png")

    st.image("images/age_distribution.png")

with tab2:
    st.subheader("Enter Passenger Details")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 30)
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare Paid", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

    sex_val = 0 if sex == "male" else 1
    embarked_val = {"S": 0, "C": 1, "Q": 2}[embarked]

    input_df = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]],
                             columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.success(f"Survived  (probability: {proba:.2%})")
        else:
            st.error(f"Did Not Survive  (probability of survival: {proba:.2%})")