import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Customer Spending Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .high-spender {
        background-color: rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
    }
    .low-spender {
        background-color: rgba(255, 152, 0, 0.2);
        border: 1px solid #FF9800;
    }
    .slider-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #424242;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .sidebar-content {
        padding: 1rem;
        background-color: #f5f5f5;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


def progress_animation():
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("Analysis complete!")
    return


st.markdown("<h1 class='main-header'>✨ Customer Spending Predictor ✨</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn.pixabay.com/photo/2017/07/01/14/04/dollar-2461576_1280.png", width=150)
    st.markdown("<h3>About this App</h3>", unsafe_allow_html=True)
    
    with st.expander("📊 What does this app do?"):
        st.write("""
        This application uses machine learning to predict whether a customer 
        is likely to be a high spender based on demographic and behavioral data.
        """)
    
    with st.expander("🔍 How to use"):
        st.write("""
        1. Adjust the input parameters using the sliders
        2. Click 'Predict' to see the analysis
        3. Review the visualizations and prediction results
        """)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9e9e9e;'>Developed with ❤️ by Avi:)</p>", unsafe_allow_html=True)


left_col, right_col = st.columns([1, 1])

# Load model and scaler
try:
    model = tf.keras.models.load_model('model.h5')
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    
    with left_col:
        st.markdown("<h2 class='sub-header'>Customer Information</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='slider-label'>Gender</div>", unsafe_allow_html=True)
        gender = st.selectbox('', ['Male', 'Female'], key='gender')
        
        st.markdown("<div class='slider-label'>Age</div>", unsafe_allow_html=True)
        age = st.slider('', 18, 70, 30, key='age')
        
        st.markdown("<div class='slider-label'>Annual Income (k$)</div>", unsafe_allow_html=True)
        income = st.slider('', 10, 150, 50, key='income')
        
        st.markdown("<div class='slider-label'>Spending Score (1-100)</div>", unsafe_allow_html=True)
        spending_score = st.slider('', 1, 100, 50, key='spending')
        
        
        gender_numeric = 1 if gender == 'Male' else 0
        
       
        input_data = np.array([[gender_numeric, age, income, spending_score]])
        
        
        predict_button = st.button('Predict', key='predict_button', 
                                  help='Click to analyze customer data')
    
    
    with right_col:
        st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
        
        if predict_button:
            
            progress_animation()
            
            try:
                # Scale input
                input_scaled = scaler.transform(input_data)
                
                # Predict
                prediction_proba = model.predict(input_scaled)[0][0]
                prediction = 1 if prediction_proba > 0.5 else 0
                
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(prediction_proba * 100),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Spending Probability"},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                        'bar': {'color': "#1E88E5"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(255, 152, 0, 0.3)'},
                            {'range': [50, 100], 'color': 'rgba(76, 175, 80, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                
                if prediction == 1:
                    st.markdown("""
                    <div class='prediction-box high-spender'>
                        <h3 style='color: #4CAF50; text-align: center;'>This customer is likely a HIGH SPENDER 💰</h3>
                        <p style='text-align: center;'>This customer has a strong tendency to spend more than average.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='prediction-box low-spender'>
                        <h3 style='color: #FF9800; text-align: center;'>This customer is NOT likely a High Spender</h3>
                        <p style='text-align: center;'>This customer tends to be more conservative with spending.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                
                st.markdown("<h3 class='sub-header'>Customer Profile</h3>", unsafe_allow_html=True)
                
                
                categories = ['Gender', 'Age', 'Income', 'Spending Score']
                
                
                normalized_values = [
                    gender_numeric * 100,  
                    (age - 18) / (70 - 18) * 100,  
                    (income - 10) / (150 - 10) * 100, 
                    spending_score  
                ]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=categories,
                    fill='toself',
                    name='Customer Profile',
                    line_color='#1E88E5'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    height=300,
                    margin=dict(l=80, r=80, t=20, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during data transformation: {str(e)}")
                st.write("Input data shape:", input_data.shape)
                st.write("Input data:", input_data)
        else:
            
            st.markdown("""
            <div style='background-color: #f5f5f5; padding: 2rem; border-radius: 10px; text-align: center;'>
                <img src="https://cdn.pixabay.com/photo/2024/02/26/14/13/shopping-8598070_1280.jpg" width="300">
                <p style='color: #757575; margin-top: 1rem;'>
                    Adjust the parameters and click 'Predict' to see the analysis results
                </p>
            </div>
            """, unsafe_allow_html=True)

 
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Understanding the Results</h2>", unsafe_allow_html=True)
    
    with st.expander("📈 What makes a high spender?"):
        st.write("""
        High spenders typically:
        - Have higher income levels
        - Show higher engagement with products/services
        - Demonstrate specific purchase patterns
        - May fall into specific age demographics
        
        Our model analyzes these factors to determine the likelihood of a customer being a high spender.
        """)
    
    with st.expander("💡 How to use this information"):
        st.write("""
        - **Marketing teams**: Target high-value customers with premium offerings
        - **Product development**: Create products that appeal to high spenders
        - **Customer service**: Provide enhanced services to high-value customers
        - **Business strategy**: Make data-driven decisions to maximize revenue
        """)

except Exception as e:
    st.error(f"Error loading model or scaler: {str(e)}")
    st.markdown("""
    <div style='background-color: #ffebee; padding: 1rem; border-radius: 5px; border: 1px solid #ef5350;'>
        <h3 style='color: #c62828;'>Model Loading Error</h3>
        <p>Please make sure the model files (model.h5 and scaler.pkl) are in the current directory.</p>
        <p>Try running the save_scaler.py script to generate a valid scaler file.</p>
    </div>
    """, unsafe_allow_html=True)