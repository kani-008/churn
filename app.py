import streamlit as st
import joblib
import pandas as pd

def main():
    st.title("Customer Churn Prediction")

    model = joblib.load("app/model.pkl")

    st.write("### Enter Customer Data")
    feature_1 = st.number_input("Feature 1")
    feature_2 = st.number_input("Feature 2")
    feature_3 = st.number_input("Feature 3")
    # Add more fields matching the feature count used during training

    if st.button("Predict"):
        input_data = pd.DataFrame([[feature_1, feature_2, feature_3]],
                                  columns=["Feature_1", "Feature_2", "Feature_3"])
        prediction = model.predict(input_data)
        result = "Churn" if prediction[0] == 1 else "Not Churn"
        st.success(f"Prediction: {result}")

if __name__ == '__main__':
    main()