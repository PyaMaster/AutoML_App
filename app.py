import os.path

import pycaret.regression
import streamlit as st
import pandas as pd

# Import profiling capability
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

# ML stuff
import pycaret.regression
import pycaret.classification
import pycaret.clustering

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Ydata Profiling and "
            "PyCaret. ")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_df = ProfileReport(df, minimal=False, orange_mode=False, explorative=True)
    st_profile_report(profile_df, navbar=True)

if choice == "ML":
    st.title("Machine Learning go")
    target = st.selectbox("Select your Target", df.columns)
    model_choice = st.radio("Select the type of model to run", ["Regression", "Classification"])
    if st.button("Run Modelling"):
        if model_choice == "Regression":
            pycaret.regression.setup(df, target=target)
            setup_df = pycaret.regression.pull()
            st.info("This is the ML Experiment settings")
            st.dataframe(setup_df)
            best_model = pycaret.regression.compare_models()
            compare_df = pycaret.regression.pull()
            st.info("This the ML Model")
            st.dataframe(compare_df)
            pycaret.regression.save_model(best_model, 'best_model')
        if model_choice == "Classification":
            pycaret.classification.setup(df, target=target)
            setup_df = pycaret.classification.pull()
            st.info("This is the ML Experiment settings")
            st.dataframe(setup_df)
            best_model = pycaret.classification.compare_models()
            compare_df = pycaret.classification.pull()
            st.info("This the ML Model")
            st.dataframe(compare_df)
            pycaret.classification.save_model(best_model, 'best_model')
        # if model_choice == "Clustering":
        #     pycaret.clustering.setup(df, target=target)
        #     setup_df = pycaret.clustering.pull()
        #     st.info("This is the ML Experiment settings")
        #     st.dataframe(setup_df)
        #     best_model = pycaret.clustering.compare_models()
        #     compare_df = pycaret.clustering.pull()
        #     st.info("This the ML Model")
        #     st.dataframe(compare_df)
        #     pycaret.clustering.save_model(best_model, 'best_model')

if choice == "Download":
    st.title("Download Your Model")
    st.info("Download the best model base on the performance of difference model of the same categories")
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
