

import streamlit as st
import pandas as pd
from joblib import load
import os

# Load the dataset for display
data_path = 'spg.csv'
dataset = pd.read_csv(data_path)

# Load the trained model
model = load("solar.pkl")

# Streamlit app
st.set_page_config(page_title="Solar Energy Generation Prediction", layout="wide")
st.title("☀️ Solar Energy Generation Prediction 🌍")

# Change background color for the page
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f9;
        color: #333333;
    }
    .stButton button {
        background-color: #FF5733;
        color: white;
    }
    .stButton button:hover {
        background-color: #ff704d;
    }
    .stMarkdown {
        background-color: #e1f5fe;
        border-radius: 10px;
        padding: 15px;
        color: #0277bd;
    }
    h1 {
        color: #2C6B51;
    }
    h2 {
        color: #388E3C;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Add a decorative image
st.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
        <img src="https://media.giphy.com/media/c3HczzhNAceOI/giphy.gif?cid=790b7611hxhebi0nsld8h686p0zsgdhr0rnysxp71n73olyr&ep=v1_gifs_search&rid=giphy.gif&ct=g" 
        alt="☀️ Turning Sunlight into Power: Solar Energy at Its Best"!" 
        style="max-width: 100%; height: auto;"/>
        <div style="margin-top: 10px; color: #2C6B51; font-size: 20px;">⚡ Turning Sunlight into Power: Solar Energy at Its Best"!</div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("⚙️ User Input Features")
def user_input_features():
    features = {}
    for column in dataset.columns[:-1]:  # Exclude target variable
        col_min = float(dataset[column].min())
        col_max = float(dataset[column].max())
        default_value = float(dataset[column].mean())
        if dataset[column].dtype == 'int64':
            features[column] = st.sidebar.slider(
                f"🔢 {column}", int(col_min), int(col_max), int(default_value)
            )
        else:
            features[column] = st.sidebar.slider(
                f"📊 {column}", col_min, col_max, default_value
            )
    return pd.DataFrame([features])

input_df = user_input_features()

# Display input features
st.subheader("📥 User Input Features")
st.write(input_df)

#button
# Predictions
if st.button("🔮 Predict Solar Power Output"):
    prediction = model.predict(input_df)
    st.subheader("⚡ Predicted Solar Power Output (kW)")
    st.markdown(f"<h2 style='color:green;'>{prediction[0]:.2f} kW</h2>", unsafe_allow_html=True)
# Predictions
# if st.button("🔮 Predict Solar Power Output"):
#     prediction = model.predict(input_df)
#     st.subheader("⚡ Predicted Solar Power Output (kW)")
#     st.markdown(f"<h2 style='color:#388E3C;'>{prediction[0]:.2f} kW</h2>", unsafe_allow_html=True)

# Dataset Overview
st.subheader("📋 Dataset Overview")
st.dataframe(dataset.head())

# Model Performance Metrics
st.subheader("📈 Model Performance")
performance = {
    "Random Forest": {"Accuracy": 0.81},
}
st.write(pd.DataFrame(performance).T)




# Importance of Solar Energy
st.subheader("🌟 Importance of Solar Energy")
st.markdown(
    """
    - 🌞 **Renewable and Sustainable**: Solar energy is a renewable and sustainable energy source.
    - 🌍 **Environmental Benefits**: It helps reduce greenhouse gas emissions and combat climate change.
    - 💡 **Cost Efficiency**: Solar power generation has minimal operating costs.
    - 🔋 **Energy Independence**: Encourages energy independence and reduces reliance on fossil fuels.
    """, 
    unsafe_allow_html=True
)
