# AutoStreamML

AutoStreamML is a web application built with Streamlit, Ydata Profiling, and PyCaret for creating an automated ML pipeline.

## Features

- **Upload**: Upload your dataset for modeling.
- **Profiling**: Conduct automated exploratory data analysis using Ydata Profiling.
- **ML**: Build and compare machine learning models using PyCaret.
- **Download**: Download the best-performing model.

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AutoStreamML.git
cd AutoStreamML

2. Install the required packages:

```bash
pip install -r requirements.txt

3. Run the application:

```bash
streamlit run app.py

## Usage

- Upload your dataset in the "Upload" section.
- Explore your data with automated EDA in the "Profiling" section.
- Build and compare machine learning models in the "ML" section.
- Download the best model in the "Download" section.

## Note

- The application stores the uploaded dataset as "sourcedata.csv".
- The best model is saved as "best_model.pkl".
