# Import necessary libraries
import os.path
import pandas as pd
import pycaret.regression
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

# Set up Streamlit sidebar with navigation options
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info(
        "This application allows you to build an automated ML pipeline using Streamlit, Ydata Profiling, and PyCaret.")

# Load dataset if it exists
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Handling different user choices
if choice == "Upload":
    # Upload and display the dataset
    st.title("Upload Your Data for Modeling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

elif choice == "Profiling":
    # Perform automated Exploratory Data Analysis (EDA)
    st.title("Automated Exploratory Data Analysis")
    profile_df = ProfileReport(df, minimal=False, orange_mode=False, explorative=True)
    st_profile_report(profile_df, navbar=True)

elif choice == "ML":
    # Build and compare machine learning models
    st.title("Machine Learning Pipeline")
    target = st.selectbox("Select your Target", df.columns)
    model_choice = st.radio("Select the type of model to run", ["Regression", "Classification"])

    if st.button("Run Modeling"):
        if model_choice == "Regression":
            # Regression model setup and training
            pycaret.regression.setup(df, target=target)
            setup_df = pycaret.regression.pull()
            st.info("This is the ML Experiment settings")
            st.dataframe(setup_df)
            best_model = pycaret.regression.compare_models()
            compare_df = pycaret.regression.pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            pycaret.regression.save_model(best_model, 'best_model')

        elif model_choice == "Classification":
            # Classification model setup and training
            pycaret.classification.setup(df, target=target)
            setup_df = pycaret.classification.pull()
            st.info("This is the ML Experiment settings")
            st.dataframe(setup_df)
            best_model = pycaret.classification.compare_models()
            compare_df = pycaret.classification.pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            pycaret.classification.save_model(best_model, 'best_model')

        # Commented out code for clustering, not yet implemented
        # elif model_choice == "Clustering":
        #     pycaret.clustering.setup(df, target=target)
        #     setup_df = pycaret.clustering.pull()
        #     st.info("This is the ML Experiment settings")
        #     st.dataframe(setup_df)
        #     best_model = pycaret.clustering.compare_models()
        #     compare_df = pycaret.clustering.pull()
        #     st.info("This is the ML Model")
        #     st.dataframe(compare_df)
        #     pycaret.clustering.save_model(best_model, 'best_model')

elif choice == "Download":
    # Provide an option to download the best model
    st.title("Download Your Model")
    st.info("Download the best model based on the performance of different models of the same category")
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")

