import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_performance_model.pkl")
preprocessor = joblib.load("student_performance_preprocessor.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 30px;
}

.result {
    text-align: center;
    padding: 30px;
    border-radius: 18px;
    margin-top: 25px;
    background-color: #1e1e1e;
    border: 1px solid #444444;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4);
}

.result h2 {
    color: #ffffff;
    margin-bottom: 10px;
}

.result h1 {
    color: #00d4ff;
    font-size: 50px;
    margin: 10px 0;
}

.result h3 {
    color: #ffffff;
    margin-top: 10px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🎓 Student Performance Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict a student\'s expected Math Score using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

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

if st.button("🔮 Predict Math Score", use_container_width=True):

    new_student = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parental_education],
        "lunch": [lunch],
        "test preparation course": [test_preparation],
        "reading score": [reading_score],
        "writing score": [writing_score]
    })

    new_student_processed = preprocessor.transform(new_student)

    prediction = model.predict(new_student_processed)[0]

    prediction = max(0, min(100, prediction))

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

