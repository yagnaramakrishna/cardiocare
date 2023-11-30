import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

DATASET_PATH = "reduced_data.csv"
MODEL_PATH = "best_model.pkl"

def main():

    @st.cache_data(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pd.read_csv(DATASET_PATH)
        return heart_df

    def user_input_features():

        age_mapping = {
            "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, 
            "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
            "70-74": 11, "75-79": 12, "80 or older": 13
        }

        had_angina = st.sidebar.selectbox("Have you ever had angina?", options=("No", "Yes"))
        had_stroke = st.sidebar.selectbox("Have you ever had a stroke?", options=("No", "Yes"))
        had_copd = st.sidebar.selectbox("Do you have COPD?", options=("No", "Yes"))
        had_kidney_disease = st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
        had_arthritis = st.sidebar.selectbox("Do you have arthritis?", options=("No", "Yes"))
        had_diabetes = st.sidebar.selectbox("Have you ever had diabetes?", options=("No", "Yes"))
        deaf_or_hard_of_hearing = st.sidebar.selectbox("Are you deaf or hard of hearing?", options=("No", "Yes"))
        difficulty_walking = st.sidebar.selectbox("Do you have difficulty walking?", options=("No", "Yes"))
        chest_scan = st.sidebar.selectbox("Have you had a chest scan?", options=("No", "Yes"))
        age_category_label = st.sidebar.selectbox("What is your age category?", options=list(age_mapping.keys()))
        pneumo_vax_ever = st.sidebar.selectbox("Have you ever had a pneumonia vaccine?", options=("No", "Yes"))
        

        age_category = age_mapping[age_category_label]

        features = pd.DataFrame({
            "HadAngina": [1 if had_angina == "Yes" else 0],
            "HadStroke": [1 if had_stroke == "Yes" else 0],
            "HadCOPD": [1 if had_copd == "Yes" else 0],
            "HadKidneyDisease": [1 if had_kidney_disease == "Yes" else 0],
            "HadArthritis": [1 if had_arthritis == "Yes" else 0],
            "HadDiabetes": [1 if had_diabetes == "Yes" else 0],
            "DeafOrHardOfHearing": [1 if deaf_or_hard_of_hearing == "Yes" else 0],
            "DifficultyWalking": [1 if difficulty_walking == "Yes" else 0],
            "ChestScan": [1 if chest_scan == "Yes" else 0],
            "AgeCategory": [age_category],
            "PneumoVaxEver": [1 if pneumo_vax_ever == "Yes" else 0],
            
        })

        return features
    st.set_page_config(page_title="CardioCare: Heart Disease Prediction Application", page_icon=":heart:", layout="wide")
    



    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .image-caption {
            text-align: center;
            font-style: italic;
        }
        
        .stButton > button {
            background-color: #0d6efd;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 5px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #0b5ed7;
        }
        </style>
        """, unsafe_allow_html=True)
    
        
    st.title("CardioCare: Heart Disease Prediction Application")
    st.subheader("Are you concerned about the health of your heart? This application is designed to assist you in assessing its condition!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("doc-main.png", caption="I'm here to assist you in evaluating the health of your heart! - Dr. Machine Learning", width=200)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        That's fascinating! Machine learning models, like Random Forest, can indeed provide accurate predictions for heart disease risk. In this app, it utilizes a Random Forest Model, trained on a substantial dataset of over 400,000 plus US residents from 2022. This model has been chosen because it achieves an impressive accuracy of approximately 94%.

        To predict your heart disease risk using this application, follow these simple steps:

        1. Provide the parameters that best describe your health and characteristics.
        2. Click the "Predict" button.
        3. Wait for the application to process your information and provide you with a prediction (yes or no) regarding your heart disease status.This user-friendly tool can quickly assess your heart disease risk based on the information you provide.
        
        Resources and facts: https://www.cdc.gov/heartdisease/prevention.htm
                    
        **Please note that the results from this app are not a medical diagnosis. It's essential to consult a human doctor for any health concerns.**
        """)


    st.sidebar.title("Input Features")
    input_df = user_input_features()
    

    model = pickle.load(open(MODEL_PATH, "rb"))

    if submit:
        prediction = model.predict(input_df)
        prediction_prob = model.predict_proba(input_df)



        feature_names = input_df.columns
        feature_importances = model.feature_importances_
        fig = px.bar(x=feature_names, y=feature_importances, labels={'x':'Feature', 'y':'Importance'},title="Feature Importances for Heart Disease Prediction")
        fig.update_layout(xaxis_title="Feature", yaxis_title="Importance")
        st.plotly_chart(fig)

                

        fig = px.pie(values=prediction_prob[0], names=["Healthy", "Heart Disease"],title="Heart Disease Prediction Probability")
        st.plotly_chart(fig)



        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_prob[0][1],
        title={'text': "Risk Level"},
        gauge={'axis': {'range': [None, 1]}, 'steps': [
            {'range': [0, 0.5], 'color': "lightgreen"},
            {'range': [0.5, 1], 'color': "lightcoral"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prediction_prob[0][1]}}))
        st.plotly_chart(fig)



        def plot_correlation_heatmap(df, target_name):

    
            corr = df.corr()
    
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                    labels=dict(x="Feature", y="Feature", color="Correlation"),
                    title="Feature Correlation Heatmap")
            return fig
        

        heart_df = load_dataset()       
        target_name = 'HadHeartAttack'
        fig = plot_correlation_heatmap(heart_df, target_name)
        st.plotly_chart(fig)




        def plot_correlation_bubble(df, target_name):
            corr = df.corr()[target_name].sort_values(ascending=False)
            fig = px.scatter(x=corr.index, y=corr.values, size=np.abs(corr.values),
                     labels={'x': 'Feature', 'y': 'Correlation with Heart Attack'},
                     title="Feature Correlations with Heart Attack Status")
            return fig

        fig = plot_correlation_bubble(heart_df, target_name)
        st.plotly_chart(fig)


        if prediction == 0:
            st.markdown(f"<span style='color: green'>**Your probability of having heart disease is {round(prediction_prob[0][1] * 100, 2)}%. You are healthy!**</span>", unsafe_allow_html=True)
            st.image("heart-okay.jpg", caption="**Congrats - It appears that your heart is in good condition - Dr. Machine Learning**")
        else:
            st.markdown(f"<span style='color: red'>**Your probability of having heart disease is {round(prediction_prob[0][1] * 100, 2)}%. It sounds like you are not healthy.**</span>", unsafe_allow_html=True)
            st.image("heart-bad.jpg", caption="**I'm concerned about the state of your heart, get it checked regularly with a human doctor - Dr. Machine Learning**")

if __name__ == "__main__":
    main()
