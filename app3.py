import streamlit as st
import pandas as pd
import os
import sweetviz as sv
from pycaret.classification import setup, compare_models, pull, save_model
import matplotlib
matplotlib.use('Agg')
import warnings

warnings.filterwarnings("ignore")

# Background image style
background_image_style = """
<style>
.stApp {
    background-image: url("https://img.freepik.com/free-vector/vibrant-fluid-gradient-background-with-curvy-shapes_1017-32108.jpg?size=626&ext=jpg&ga=GA1.1.2008272138.1708473600&semt=ais");
    background-size: cover;
}
</style>
"""

st.markdown(background_image_style, unsafe_allow_html=True)

# Custom CSS for headings
heading_customization = """
<style>
/* CSS for customizing all markdown headers in the app */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    font-weight: bold;
    font-size: 3.0em; /* Adjust the size as needed */
}

/* Alternatively, if you want to specifically target headers in expanders */
.stExpander .stMarkdown h1, .stExpander .stMarkdown h2, .stExpander .stMarkdown h3, 
.stExpander .stMarkdown h4, .stExpander .stMarkdown h5, .stExpander .stMarkdown h6 {
    font-weight: bold;
    font-size: 2.5em; /* Adjust the size as needed */
}
</style>
"""

st.markdown(heading_customization, unsafe_allow_html=True)
# Check if the dataset is already loaded
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Set up the title and main image URL
st.title("Welcome to the AUTOML APP FOR PREDICTIVE MAINTENANCE")
auto_url = "predictive_main.jpg"

# Home Section
with st.expander("🏠 Home"):
      
      st.image(auto_url, use_column_width=True)
      st.markdown("""
# Welcome to the AUTOML APP FOR PREDICTIVE MAINTENANCE

This application harnesses the power of Automated Machine Learning (AutoML) to enable users to predict maintenance needs based on their data. Our goal is to simplify the process of utilizing machine learning for predictive maintenance tasks, making it accessible to users with varying levels of expertise in data science.

## Key Features:

- **Data Upload:** Users can easily upload their dataset in CSV format. This data should reflect the operational metrics relevant to maintenance, such as equipment performance indicators, failure history, or other sensor readings.

- **Automated Exploratory Data Analysis:** Once your data is uploaded, the app provides an automated exploratory analysis using Sweetviz. This offers valuable insights into your data, including distribution, correlation, and missing values, helping you understand the dataset's characteristics before moving on to model training.

- **Auto Machine Learning:** With just a few clicks, users can train multiple machine learning models on their data. Our AutoML setup evaluates various models to find the one that best predicts maintenance needs. You don't need to worry about the complexities of model selection, parameter tuning, or performance evaluation; the app handles it all.

- **Model Evaluation and Download:** After training, the app displays the performance of the best model and allows users to download this model for future use. This enables easy integration of the predictive model into your maintenance planning processes.

## How to Navigate This App:

1. **Upload Your Data:** Start by uploading your dataset through the 'Upload Your Data' section. Ensure your data is in the correct format (CSV) and includes all necessary features for predictive maintenance.

2. **Explore Your Data:** Use the 'Automated Exploratory Data Analysis' section to get insights into your data's quality and characteristics. This step is crucial for understanding the dataset you're working with.

3. **Train Your Model:** Navigate to the 'Auto Machine Learning' section, select your target feature (the outcome you wish to predict), and train your model. The app will guide you through the process.

4. **Evaluate and Download the Model:** Review the performance of the best model in the 'Download Best Model' section. You can download the model for external use.
    """, unsafe_allow_html=True)
   

with st.expander("📤 Upload Your Data"):
    data = st.file_uploader("Please, upload your dataset here", type=['csv'])
    if data:
        st.session_state.df = pd.read_csv(data)
        # No need to save the CSV here again unless it's for another purpose
        st.dataframe(st.session_state.df)

with st.expander("🔍 Automated Exploratory Data Analysis"):
    # Ensure the Sweetviz report generation is correctly using the session state
    if st.session_state.df is not None:
        report = sv.analyze(st.session_state.df)
        report.show_html(filepath='SWEETVIZ_REPORT.html', open_browser=False)
        HtmlFile = open('SWEETVIZ_REPORT.html', 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.components.v1.html(source_code, width=1000, height=500, scrolling=True)
    else:
        st.error("Please upload data in the 'Upload Your Data' section first.")

with st.expander("🤖 Auto Machine Learning"):
    if st.session_state.df is not None:
        st.subheader("AUTOMATED MACHINE LEARNING COMPUTATION")
        st.info("In this section, the app builds and trains different machine learning models with the train data. User has to ONLY identify and enter the target variable")
        st.info("NOTE: If no data is uploaded at the Data Upload Page, this page will show an error message.")
        target = st.selectbox("Please, select your target feature", st.session_state.df.columns)
        if st.button("Train model"):
            # Directly use st.session_state.df here
            setup(data=st.session_state.df, target=target, html=False, session_id=123)
            best_model = compare_models()
            # Use pull() to get a dataframe of the models' performance
            compare_df = pull()
            st.info("This is the performance of the machine learning models:")
            st.dataframe(compare_df)
            # Save and display the best model
            save_model(best_model, "best_model.pkl")
            st.write("Best Model:", best_model)
            
    else:
        st.info("Please upload data in the 'Upload Your Data' section.")

# Model Download Section
with st.expander("💾 Download Best Model"):
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", 'rb') as f:
            st.download_button("Download Best Model", f, "best_model.pkl")
    else:
        st.error("No model has been trained and saved yet.")