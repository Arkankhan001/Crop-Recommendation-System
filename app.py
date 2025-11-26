# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests

# ------------- CONFIG -----------------

MODEL_PATH = "crop_recommender.pkl"
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Put your OpenWeather API key here if you want auto-fill:
OPENWEATHER_API_KEY = ""  # e.g. "your_api_key_here"

# Simple fertilizer suggestions (you can expand this dict)
FERTILIZER_MAP = {
    "rice": "Use NPK 10-26-26 and apply urea in split doses. Maintain standing water.",
    "maize": "Balanced NPK with higher nitrogen. Use urea + DAP, avoid waterlogging.",
    "chickpea": "Less nitrogen needed (it fixes N). Use SSP and well-decomposed FYM.",
    "banana": "High K demand. Use NPK 8-10-22 and apply organic compost regularly.",
}

# ------------- HELPER FUNCTIONS -----------------


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_weather(city_name: str, api_key: str):
    """
    Fetch temperature (°C) and humidity (%) from OpenWeather.
    Returns (temperature, humidity) or (None, None) on error.
    """
    try:
        if not api_key:
            return None, None

        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city_name}&appid={api_key}"
        )
        res = requests.get(url, timeout=5)
        data = res.json()

        if data.get("cod") != 200:
            return None, None

        temp_k = data["main"]["temp"]
        temp_c = temp_k - 273.15
        humidity = data["main"]["humidity"]
        return round(temp_c, 2), float(humidity)
    except Exception:
        return None, None


def get_fertilizer_advice(crop_name: str) -> str:
    crop_name = crop_name.lower()
    for key in FERTILIZER_MAP:
        if key in crop_name:
            return FERTILIZER_MAP[key]
    return "Use balanced NPK fertilizer and add organic compost to improve soil health."


# ------------- UI LAYOUT -----------------

st.set_page_config(
    page_title="SmartCrop AI - Crop Recommendation",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    "<h1 style='text-align:center;'>🌾 SmartCrop AI – Crop Recommendation System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Make data-driven decisions for better yield and profit.</p>",
    unsafe_allow_html=True,
)

model = load_model()

left, right = st.columns([2, 1])

# -------- LEFT: Input form --------
with left:
    st.subheader("Enter Soil & Weather Parameters")

    use_weather = st.checkbox(
        "Auto-fill temperature & humidity using city name (OpenWeather API)",
        value=False,
    )

    if use_weather:
        city = st.text_input("City / Village name (for weather API)", value="Mehsana")
        if OPENWEATHER_API_KEY:
            temp_api, hum_api = get_weather(city, OPENWEATHER_API_KEY)
            if temp_api is not None:
                st.info(f"From API → Temperature: {temp_api} °C, Humidity: {hum_api} %")
            else:
                st.warning("Could not fetch weather. Check city name or API key.")
        else:
            st.warning("Add your OpenWeather API key in the code to use this feature.")
            temp_api, hum_api = None, None
    else:
        temp_api, hum_api = None, None

    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=temp_api if temp_api is not None else 25.0,
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=hum_api if hum_api is not None else 60.0,
    )
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    if st.button("🔍 Recommend Best Crop"):
        user_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        predicted_crop = model.predict(user_data)[0]

        st.success(f"✅ Recommended Crop: **{predicted_crop}**")

        # Fertilizer advice
        fertilizer_text = get_fertilizer_advice(predicted_crop)
        st.markdown("### 🌱 Fertilizer Recommendation")
        st.write(fertilizer_text)

# -------- RIGHT: Model insights --------
with right:
    st.subheader("📊 Model Insights")

    # Feature importance from RandomForest inside pipeline
    rf_model = model.named_steps["model"]
    importances = rf_model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances
    }).sort_values("importance", ascending=False)

    st.markdown("**Feature Importance** (higher = more impact):")
    st.bar_chart(imp_df.set_index("feature"))

    st.markdown("---")
    st.markdown("**How it works:**")
    st.write(
        """
        - Uses a tuned Random Forest model with cross-validation  
        - Considers N, P, K, temperature, humidity, pH, and rainfall  
        - Outputs the most suitable crop and a basic fertilizer suggestion  
        """
    )
