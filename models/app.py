import os
import streamlit as st
from PIL import Image

# Import models
from predict import predict_disease
from fertilizer import predict_fertilizer
from yield_prediction import predict_yield, get_categories
from weather import predict_weather

# Page configuration
st.set_page_config(
    page_title="AgriAI - Smart Agriculture Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI CSS Injection
st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main gradient header */
    .header-container {
        background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
        color: white;
        padding: 35px 25px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }
    .header-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .header-subtitle {
        font-size: 1.15rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Cards */
    .agro-card {
        background-color: white;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #f1f3f5;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .agro-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        border-color: #71b280;
    }
    
    /* Stats badge */
    .stat-badge {
        font-size: 2.2rem;
        font-weight: 700;
        color: #134e5e;
        margin-bottom: 5px;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Recommendation Alert Box */
    .rec-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 5px solid #0284c7;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        font-size: 1.05rem;
    }
    .rec-box-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #16a34a;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        font-size: 1.05rem;
    }
    
    /* Disease diagnostic card */
    .disease-res-card {
        padding: 20px;
        border-radius: 12px;
        margin-top: 10px;
        border: 1px solid #ffccd5;
        background-color: #fff5f5;
    }
    .disease-res-healthy {
        padding: 20px;
        border-radius: 12px;
        margin-top: 10px;
        border: 1px solid #dcfce7;
        background-color: #f0fdf4;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation styling
st.sidebar.markdown("""
<div style='text-align: center; padding-bottom: 20px;'>
    <h2 style='color:#134e5e; font-weight: 700;'>🌱 AgriAI Platform</h2>
    <p style='color:#6c757d; font-size: 0.9rem;'>Intelligence for Sustainable Farming</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Selection Menu
menu = st.sidebar.radio(
    "Navigation",
    ["🏡 Overview & Dashboard", "🍃 Leaf Disease Diagnosis", "🧪 Fertilizer Advisor", "🌾 Crop Yield Projections", "🌦️ Weather Analyzer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size: 0.85rem; color: #6c757d;'>
    <b>System Status:</b> Operational <span style='color: #16a34a;'>●</span><br>
    <b>Model Modules:</b> Loaded ✅<br>
    <b>Platform Engine:</b> v1.0.0
</div>
""", unsafe_allow_html=True)


# ==========================================
# 1. OVERVIEW PAGE
# ==========================================
if menu == "🏡 Overview & Dashboard":
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Welcome to AgriAI Analytics Platform</div>
        <div class="header-subtitle">Empowering farmers with state-of-the-art AI-driven insights for disease detection, soil nutrition, and yield forecasting.</div>
    </div>
    """, unsafe_allow_html=True)

    # Core Stats Dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="agro-card">
            <div class="stat-badge">78</div>
            <div class="stat-label">Supported Leaf Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="agro-card">
            <div class="stat-badge">246k+</div>
            <div class="stat-label">Yield Production Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="agro-card">
            <div class="stat-badge">96k+</div>
            <div class="stat-label">Weather Log Data</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="agro-card">
            <div class="stat-badge">4 Models</div>
            <div class="stat-label">Active Predictive Modules</div>
        </div>
        """, unsafe_allow_html=True)

    # Main platform description
    st.subheader("Interactive Agricultural Workflow")
    st.markdown("""
    Select a module from the left sidebar navigation panel to access platform services:
    """)
    
    flow_col1, flow_col2, flow_col3 = st.columns(3)
    with flow_col1:
        st.markdown("""
        <div class="agro-card" style="height: 250px;">
            <h4 style="color:#134e5e;">🍃 Disease Diagnosis</h4>
            <p style="color:#495057; font-size: 0.95rem;">
                Upload close-up photos of plant leaves or select sample images. The Convolutional Neural Network (CNN) detects diseases instantly, allowing early intervention to protect crop yield.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with flow_col2:
        st.markdown("""
        <div class="agro-card" style="height: 250px;">
            <h4 style="color:#134e5e;">🧪 Fertilizer Optimization</h4>
            <p style="color:#495057; font-size: 0.95rem;">
                Input soil test readings (NPK levels, soil moisture) and climate parameters. The model matches conditions against agronomic guidelines to recommend exact fertilizer requirements.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with flow_col3:
        st.markdown("""
        <div class="agro-card" style="height: 250px;">
            <h4 style="color:#134e5e;">🌾 Yield & Climate Analytics</h4>
            <p style="color:#495057; font-size: 0.95rem;">
                Evaluate district-level crop trends based on historical outputs. Combine crop inputs and local sensor readings to estimate agricultural outputs and classify weather statuses.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 2. LEAF DISEASE DIAGNOSIS PAGE
# ==========================================
elif menu == "🍃 Leaf Disease Diagnosis":
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Leaf Disease Diagnostic Lab</div>
        <div class="header-subtitle">Upload high-resolution images of crop leaves to classify anomalies and identify optimal treatments.</div>
    </div>
    """, unsafe_allow_html=True)

    # Define test images directory
    TEST_DIR = "plant_disease/test/test"
    test_images = []
    if os.path.exists(TEST_DIR):
        test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Dynamic columns
    col_input, col_display = st.columns([1, 1])

    with col_input:
        st.subheader("Image Input Selection")
        
        # Interactive Image Selection modes
        input_mode = st.radio("Choose Input Method:", ["Upload Custom Leaf Image", "Select from Test Gallery"])
        
        selected_image_path = None
        
        if input_mode == "Upload Custom Leaf Image":
            uploaded_file = st.file_uploader("Upload leaf image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Save temp image
                os.makedirs("models/temp", exist_ok=True)
                selected_image_path = "models/temp/temp_uploaded.jpg"
                img = Image.open(uploaded_file)
                img.save(selected_image_path)
        else:
            if test_images:
                selected_test_file = st.selectbox(
                    "Select a sample image from the test set:",
                    test_images,
                    format_func=lambda x: " ".join(x.replace(".JPG", "").replace(".jpg", "").split())
                )
                selected_image_path = os.path.join(TEST_DIR, selected_test_file)
            else:
                st.warning("Test gallery folder not found.")
        
        # Diagnostic button execution
        if selected_image_path:
            st.markdown("<br>", unsafe_allow_html=True)
            diagnose_btn = st.button("🚀 Run Disease Diagnosis", use_container_width=True)

    with col_display:
        st.subheader("Leaf Preview & Diagnosis")
        if selected_image_path:
            st.image(selected_image_path, caption="Active Image Preview", use_container_width=True)
            
            if 'diagnose_btn' in locals() and diagnose_btn:
                with st.spinner("Processing image and running neural net classification..."):
                    try:
                        label, confidence, raw_label = predict_disease(selected_image_path)
                        
                        is_healthy = "healthy" in label.lower()
                        
                        if is_healthy:
                            st.markdown(f"""
                            <div class="disease-res-healthy">
                                <h3 style="color:#16a34a; margin-top:0;">🍃 Healthy Plant Detected</h3>
                                <p><b>Status Class:</b> {label}</p>
                                <p><b>Confidence Rating:</b> {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class="rec-box-success">
                                <b>Recommendation:</b><br>
                                The leaf matches profiles of a healthy specimen. No signs of infection found. Continue optimal watering and nutrient regimes.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="disease-res-card">
                                <h3 style="color:#dc2626; margin-top:0;">⚠️ Disease Detected</h3>
                                <p><b>Identified Issue:</b> {label}</p>
                                <p><b>Confidence Rating:</b> {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Simple contextual recommendation rules
                            crop = label.split("-")[0].strip() if "-" in label else label
                            st.markdown(f"""
                            <div class="rec-box">
                                <b>Treatment Guidelines:</b><br>
                                <ul>
                                    <li>Isolate infected plants to prevent spores from spreading.</li>
                                    <li>Apply targeted organic copper fungicides or recommended bio-pesticides.</li>
                                    <li>Prune highly affected branches and avoid overhead sprinkling irrigation to reduce humidity on leaves.</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during model inference: {e}")
        else:
            st.info("Please select or upload an image to activate diagnostic tools.")


# ==========================================
# 3. FERTILIZER ADVISOR PAGE
# ==========================================
elif menu == "🧪 Fertilizer Advisor":
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Soil Nutrient & Fertilizer Advisor</div>
        <div class="header-subtitle">Input physical parameters of the soil and environment to calculate fertilizer recommendation values.</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([4, 3])

    with col_form:
        st.subheader("Environmental & Soil Inputs")
        
        # Grid input
        with st.form("fertilizer_form"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
                crop_type = st.selectbox("Crop Target", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"])
                temp = st.slider("Temperature (°C)", 10, 50, 28)
                humidity = st.slider("Environmental Humidity (%)", 10, 100, 55)
            with f_col2:
                moisture = st.slider("Soil Moisture Level (cbar)", 10, 100, 42)
                n_val = st.number_input("Nitrogen (N) Content (mg/kg)", 0, 150, 22)
                p_val = st.number_input("Phosphorous (P) Content (mg/kg)", 0, 150, 20)
                k_val = st.number_input("Potassium (K) Content (mg/kg)", 0, 150, 12)
            
            submit_fert = st.form_submit_button("Calculate Recommended Fertilizer", use_container_width=True)

    with col_result:
        st.subheader("Recommendation Result")
        if submit_fert:
            with st.spinner("Analyzing soil metrics using Random Forest classifier..."):
                try:
                    prediction, confidence = predict_fertilizer(
                        temp, humidity, moisture, soil_type, crop_type, n_val, k_val, p_val
                    )
                    
                    st.markdown(f"""
                    <div class="agro-card" style="text-align: center; border-left: 6px solid #134e5e;">
                        <h4 style="margin: 0; color: #6c757d; text-transform: uppercase;">Best Fit Fertilizer</h4>
                        <div class="stat-badge" style="font-size: 3rem; margin: 15px 0; color: #134e5e;">{prediction}</div>
                        <p style="font-size: 1.1rem; font-weight: 600; color: #2e7d32;">Confidence Rating: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Educational insights based on NPK
                    st.markdown(f"""
                    <div class="rec-box-success">
                        <b>Nutrient Analysis for {crop_type}:</b><br>
                        Applying <b>{prediction}</b> will adjust the soil ratio to meet the specific physiological requirements of {crop_type} under your environment ({temp}°C, {humidity}% humidity).
                        Ensure soil moisture is maintained at optimal rates during fertilizer application.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Inference pipeline error: {e}")
        else:
            st.info("Input soil metrics and submit the form to review fertilizer recommendations.")


# ==========================================
# 4. CROP YIELD PROJECTIONS PAGE
# ==========================================
elif menu == "🌾 Crop Yield Projections":
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Crop Yield & Production Forecasting</div>
        <div class="header-subtitle">Evaluate historical crop production standards and project yields based on cultivation areas.</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        # Load unique options dynamically
        cats = get_categories()
        
        col_y_inputs, col_y_outputs = st.columns([1, 1])
        
        with col_y_inputs:
            st.subheader("Agricultural Parameters")
            
            # Nested dropdowns: State selects District
            # Filter states to requested South Indian states
            allowed_states = ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana"]
            filtered_states = [s for s in cats['states'] if s in allowed_states]
            state = st.selectbox("Select State Name", filtered_states)
            
            # Fetch districts dynamically filtered by selected state
            districts_available = cats['state_districts'].get(state, [])
            district = st.selectbox("Select District Name", districts_available)
            
            y_col1, y_col2 = st.columns(2)
            with y_col1:
                crop = st.selectbox("Crop Type", cats['crops'])
                season = st.selectbox("Season", cats['seasons'])
            with y_col2:
                year = st.number_input("Year of Cultivation", 1990, 2030, 2026)
                area = st.number_input("Cultivation Area (Hectares)", 0.1, 1000000.0, 100.0)
            
            submit_yield = st.button("Predict Production & Projections", use_container_width=True)

        with col_y_outputs:
            st.subheader("Yield Forecast Summary")
            if submit_yield:
                with st.spinner("Processing regression trees..."):
                    try:
                        production, yield_metric = predict_yield(
                            state, district, year, season, crop, area
                        )
                        
                        st.markdown(f"""
                        <div class="agro-card" style="border-left: 6px solid #71b280;">
                            <h4 style="color:#6c757d; margin-top:0;">Estimated Production</h4>
                            <div class="stat-badge" style="color: #2e7d32;">{production:,.2f} <span style="font-size: 1.5rem; font-weight: 400; color: #6c757d;">Metric Tons</span></div>
                            <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                            <h4 style="color:#6c757d;">Projected Soil Yield</h4>
                            <div class="stat-badge" style="color: #134e5e;">{yield_metric:.4f} <span style="font-size: 1.5rem; font-weight: 400; color: #6c757d;">Tons / Hectare</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="rec-box">
                            <b>Regional Context ({district}, {state}):</b><br>
                            For {crop} planted during the {season} season, a cultivation size of {area:,.1f} hectares is projected to produce {production:,.1f} tons. 
                            This output is calculated based on historical yield performance within this district.
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction logic error: {e}")
            else:
                st.info("Input land size and location details to generate yield forecast metrics.")
    except Exception as e:
        st.error(f"Could not load yield categorization metadata. Ensure model resources are built correctly. Error: {e}")


# ==========================================
# 5. WEATHER ANALYZER PAGE
# ==========================================
elif menu == "🌦️ Weather Analyzer":
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Meteorological Classifier</div>
        <div class="header-subtitle">Evaluate meteorological sensors to classify overall weather conditions.</div>
    </div>
    """, unsafe_allow_html=True)

    w_col1, w_col2 = st.columns([1, 1])

    with w_col1:
        st.subheader("Atmospheric Measurements")
        with st.form("weather_form"):
            temp = st.slider("Temperature (C)", -30.0, 50.0, 15.0)
            app_temp = st.slider("Apparent Feel Temperature (C)", -30.0, 50.0, 14.0)
            humidity = st.slider("Humidity Ratio (0-1)", 0.0, 1.0, 0.7)
            
            w_subcol1, w_subcol2 = st.columns(2)
            with w_subcol1:
                wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 12.0)
                wind_bearing = st.number_input("Wind Bearing (Degrees)", 0.0, 360.0, 180.0)
            with w_subcol2:
                visibility = st.number_input("Visibility Range (km)", 0.0, 30.0, 10.0)
                pressure = st.number_input("Barometric Pressure (mb)", 900.0, 1100.0, 1013.25)
            
            submit_weather = st.form_submit_button("Classify Atmospheric Condition", use_container_width=True)

    with w_col2:
        st.subheader("Weather Assessment")
        if submit_weather:
            with st.spinner("Classifying weather metrics..."):
                try:
                    summary, confidence = predict_weather(
                        temp, app_temp, humidity, wind_speed, wind_bearing, visibility, pressure
                    )
                    
                    # Choose a matching emoji for visual interest
                    weather_emoji = "🌤️"
                    lower_summary = summary.lower()
                    if "clear" in lower_summary:
                        weather_emoji = "☀️"
                    elif "cloudy" in lower_summary:
                        weather_emoji = "☁️"
                    elif "overcast" in lower_summary:
                        weather_emoji = "☁️"
                    elif "foggy" in lower_summary or "haze" in lower_summary:
                        weather_emoji = "🌫️"
                    elif "rain" in lower_summary or "drizzle" in lower_summary:
                        weather_emoji = "🌧️"
                    elif "snow" in lower_summary:
                        weather_emoji = "❄️"
                    elif "breezy" in lower_summary or "windy" in lower_summary:
                        weather_emoji = "💨"
                        
                    st.markdown(f"""
                    <div class="agro-card" style="text-align: center; border-left: 6px solid #134e5e;">
                        <h4 style="margin: 0; color: #6c757d; text-transform: uppercase;">Predicted Weather Summary</h4>
                        <div style="font-size: 5rem; margin: 10px 0;">{weather_emoji}</div>
                        <div class="stat-badge" style="font-size: 2.2rem; color: #134e5e; margin-bottom: 10px;">{summary}</div>
                        <p style="font-size: 1.1rem; font-weight: 600; color: #2e7d32;">Confidence Rating: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Inference error: {e}")
        else:
            st.info("Input atmospheric parameters and execute to predict summary values.")
