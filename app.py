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

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .result {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }

    .result h1 {
        font-size: 45px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    "<h1>🎓 Student Performance Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p>Predict a student's expected Math Score using Machine Learning</p>",
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
if st.button(
    "🔮 Predict Math Score",
    use_container_width=True
):

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
