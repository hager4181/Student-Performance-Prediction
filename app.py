import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("student_performance_model.pkl")
preprocessor = joblib.load("student_performance_preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Dark Mode CSS
st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0e1117;
        color: #ffffff;
    }

    /* Main content */
    .main {
        background: #0e1117;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #a8b3c7;
        margin-bottom: 30px;
    }

    /* Section headers */
    h2, h3 {
        color: #ffffff !important;
    }

    /* Labels */
    label {
        color: #e6e6e6 !important;
    }

    /* Select boxes and number inputs */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #1b1f2a !important;
        color: #ffffff !important;
        border: 1px solid #3a4252 !important;
        border-radius: 8px !important;
    }

    /* Select box text */
    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    /* Number input */
    div[data-testid="stNumberInput"] input {
        color: #ffffff !important;
    }

    /* Dropdown menu */
    div[data-baseweb="popover"] {
        background-color: #1b1f2a !important;
    }

    div[role="option"] {
        background-color: #1b1f2a !important;
        color: #ffffff !important;
    }

    div[role="option"]:hover {
        background-color: #2a3140 !important;
    }

    /* Prediction button */
    .stButton > button {
        width: 100%;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px;
        font-size: 18px;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
        border: none;
    }

    /* Prediction result */
    .result {
        background: #161b26;
        border: 1px solid #30384a;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
    }

    .result h2 {
        color: #cbd5e1 !important;
        margin-bottom: 10px;
    }

    .result h1 {
        color: #60a5fa !important;
        font-size: 52px;
        margin: 5px 0;
    }

    .result h3 {
        color: #e2e8f0 !important;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30384a;
        border-radius: 10px;
    }

    /* Info box */
    div[data-testid="stAlert"] {
        background-color: #161b26;
        border: 1px solid #30384a;
        color: #dbeafe;
    }

    /* Divider */
    hr {
        border-color: #30384a !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    '<div class="main-title">🎓 Student Performance Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict a student\'s expected Math Score using Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()

# Student information
st.header("👨‍🎓 Student Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["female", "male"]
    )

    race = st.selectbox(
        "Race / Ethnicity",
        [
            "group A",
            "group B",
            "group C",
            "group D",
            "group E"
        ]
    )

    parental_education = st.selectbox(
        "Parental Level of Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ]
    )

    lunch = st.selectbox(
        "Lunch",
        [
            "standard",
            "free/reduced"
        ]
    )

with col2:

    test_preparation = st.selectbox(
        "Test Preparation Course",
        [
            "none",
            "completed"
        ]
    )

    reading_score = st.number_input(
        "Reading Score",
        min_value=0,
        max_value=100,
        value=70
    )

    writing_score = st.number_input(
        "Writing Score",
        min_value=0,
        max_value=100,
        value=70
    )

st.divider()

# Prediction button
if st.button("🔮 Predict Math Score", use_container_width=True):

    # Create input DataFrame
    new_student = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parental_education],
        "lunch": [lunch],
        "test preparation course": [test_preparation],
        "reading score": [reading_score],
        "writing score": [writing_score]
    })

    # Apply preprocessing
    new_student_processed = preprocessor.transform(new_student)

    # Make prediction
    prediction = model.predict(new_student_processed)[0]

    # Keep prediction between 0 and 100
    prediction = max(0, min(100, prediction))

    # Performance classification
    if prediction >= 90:
        performance = "🌟 Excellent"
    elif prediction >= 75:
        performance = "🟢 Very Good"
    elif prediction >= 60:
        performance = "🟡 Good"
    elif prediction >= 50:
        performance = "🟠 Average"
    else:
        performance = "🔴 Needs Improvement"

    # Display prediction
    st.markdown(
        f"""
        <div class="result">
            <h2>🎯 Predicted Math Score</h2>
            <h1>{prediction:.2f} / 100</h1>
            <h3>{performance}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display student information
    st.subheader("📋 Student Information")

    result_data = {
        "Gender": gender,
        "Race / Ethnicity": race,
        "Parental Education": parental_education,
        "Lunch": lunch,
        "Test Preparation": test_preparation,
        "Reading Score": reading_score,
        "Writing Score": writing_score
    }

    result_df = pd.DataFrame(
        result_data.items(),
        columns=["Feature", "Value"]
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# About project
st.header("📊 About the Project")

st.write(
    """
    This project uses Machine Learning to predict a student's expected
    Math Score based on demographic information, parental education,
    lunch type, test preparation course, reading score, and writing score.
    """
)

st.info(
    "The prediction is generated using a trained Machine Learning model "
    "and the same preprocessing pipeline used during training."
)
