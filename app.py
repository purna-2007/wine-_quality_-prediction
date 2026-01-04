import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="🍷 Wine Quality Classification", page_icon="🍷", layout="centered")

# --- GOLD + CENTER BURST ANIMATION CSS ---
def set_gold_classification_style():
    st.markdown("""
    <style>
    /* --- REMOVE TOP WHITE BARS, MENU, AND DECORATION --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0rem;}
    [data-testid="stDecoration"] {display: none;}
    .stAppDeployButton {display:none;}

    /* Background Image */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.8)), 
                    url("https://thumbs.dreamstime.com/b/winter-wine-bottle-nad-glass-wallpaper-background-picture-346941239.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }

    /* Force Labels to be White */
    .stNumberInput label, div[data-testid="stWidgetLabel"] p {
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        text-shadow: 2px 2px 4px #000;
    }
    
    .success-label {
        font-size: 32px !important;
        color: #00FFCC !important;
        text-align: center;
        font-weight: bold;
        text-transform: uppercase;
        animation: bounce 1.2s infinite;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-12px); }
        60% { transform: translateY(-6px); }
    }

    /* FALLING GOLD SPARKLES - Hidden by default */
    .gold-sparkles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 9999;
    }

    .gold-particle {
        position: absolute;
        width: 15px;
        height: 15px;
        background: radial-gradient(circle, #FFD700 0%, #FFA500 50%, #FF8C00 100%);
        border-radius: 50%;
        opacity: 0;
        box-shadow: 0 0 20px #FFD700;
    }

    /* Triggered Animation */
    .active .gold-particle {
        animation: gold-fall 4s linear forwards;
    }

    @keyframes gold-fall {
        0% { transform: translateY(-10vh) scale(0); opacity: 0; }
        10% { opacity: 1; transform: translateY(0) scale(1); }
        90% { opacity: 1; }
        100% { transform: translateY(110vh) rotate(360deg) scale(0.5); opacity: 0; }
    }

    /* Position the particles randomly across the screen */
    .gold-particle:nth-child(1) { left: 10%; animation-delay: 0.1s; }
    .gold-particle:nth-child(2) { left: 20%; animation-delay: 0.5s; }
    .gold-particle:nth-child(3) { left: 35%; animation-delay: 0.2s; }
    .gold-particle:nth-child(4) { left: 50%; animation-delay: 0.8s; }
    .gold-particle:nth-child(5) { left: 65%; animation-delay: 0.3s; }
    .gold-particle:nth-child(6) { left: 80%; animation-delay: 0.6s; }
    .gold-particle:nth-child(7) { left: 90%; animation-delay: 0.4s; }
    .gold-particle:nth-child(8) { left: 25%; animation-delay: 1.0s; }
    .gold-particle:nth-child(9) { left: 75%; animation-delay: 0.7s; }
    .gold-particle:nth-child(10) { left: 45%; animation-delay: 1.2s; }

    /* CENTER BURST */
    .center-burst {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: 10000;
    }

    .center-star {
        position: absolute;
        width: 20px;
        height: 20px;
        background: #FFD700;
        border-radius: 50%;
        opacity: 0;
    }

    .active .center-star {
        animation: starBurst 1.5s ease-out forwards;
    }

    @keyframes starBurst {
        0% { opacity: 1; transform: scale(1) translate(0, 0); }
        100% { opacity: 0; transform: scale(0) translate(var(--x), var(--y)); }
    }

    /* Button Style */
    .stButton > button {
        background: linear-gradient(145deg, #722f37, #8B4513);
        color: white;
        border-radius: 15px;
        width: 100%;
        height: 3.5em;
        font-size: 20px;
        font-weight: bold;
        border: 2px solid #FFD700;
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px #FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

set_gold_classification_style()

# --- TRIGGER FUNCTION ---
def trigger_animations():
    st.markdown("""
    <div class="gold-sparkles active">
        <div class="gold-particle"></div><div class="gold-particle"></div>
        <div class="gold-particle"></div><div class="gold-particle"></div>
        <div class="gold-particle"></div><div class="gold-particle"></div>
        <div class="gold-particle"></div><div class="gold-particle"></div>
        <div class="gold-particle"></div><div class="gold-particle"></div>
    </div>
    <div class="center-burst active">
        <div class="center-star" style="--x:150px; --y:150px;"></div>
        <div class="center-star" style="--x:-150px; --y:150px;"></div>
        <div class="center-star" style="--x:150px; --y:-150px;"></div>
        <div class="center-star" style="--x:-150px; --y:-150px;"></div>
        <div class="center-star" style="--x:0px; --y:200px;"></div>
        <div class="center-star" style="--x:0px; --y:-200px;"></div>
    </div>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        rf_model = pickle.load(open('finalized_RFmodel.sav', 'rb'))
        scaler = pickle.load(open('scaler_model.sav', 'rb'))
        return rf_model, scaler
    except:
        return None, None

rf_model, scaler = load_models()

# --- UI ---
st.title("🍷 Premium Wine Quality Predictor")
st.write("Enter the wine chemical properties below to get a quality rating.")

col1, col2 = st.columns(2)
with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=8.3)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.5)
    citric_acid = st.number_input("Citric Acid", value=0.3)
    residual_sugar = st.number_input("Residual Sugar", value=2.5)
    chlorides = st.number_input("Chlorides", value=0.08)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=45.0)
    density = st.number_input("Density", value=0.99)
    pH_val = st.number_input("pH", value=3.3)
    sulphates = st.number_input("Sulphates", value=0.6)
    alcohol = st.number_input("Alcohol %", value=10.5)

if st.button("🚀 PREDICT QUALITY"):
    if rf_model is not None:
        data = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': np.log(residual_sugar) if residual_sugar > 0 else 0,
            'chlorides': np.log(chlorides) if chlorides > 0 else 0,
            'free sulfur dioxide': np.log(free_sulfur_dioxide) if free_sulfur_dioxide > 0 else 0,
            'total sulfur dioxide': np.log(total_sulfur_dioxide) if total_sulfur_dioxide > 0 else 0,
            'density': density,
            'pH': pH_val,
            'sulphates': np.log(sulphates) if sulphates > 0 else 0,
            'alcohol': alcohol
        }
        
        input_scaled = scaler.transform(pd.DataFrame([data]))
        prediction = rf_model.predict(input_scaled)[0]
        
        # This triggers the gold balls to fall ONLY NOW
        trigger_animations()
        
        st.markdown('<p class="success-label">✅ PREDICTION COMPLETE!</p>', unsafe_allow_html=True)
        
        if prediction >= 7:
            st.markdown("<h1 style='text-align: center; color: #FFD700;'>🌟 EXCELLENT QUALITY 🍷</h1>", unsafe_allow_html=True)
        elif prediction >= 6:
            st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>👍 GOOD QUALITY 🍷</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: #FFA500;'>⚠️ POOR QUALITY 🍷</h1>", unsafe_allow_html=True)
    else:
        st.error("Model not loaded.")

st.markdown("---")
st.caption("Premium Wine Quality Analysis 🍷 |")