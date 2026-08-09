# Student Performance Prediction

## Project Overview

Student Performance Prediction is a Machine Learning project that predicts a student's Math Score based on demographic information, parental education, lunch type, test preparation, reading score, and writing score.

The project uses a trained Machine Learning model and a preprocessing pipeline to make predictions through an interactive Streamlit web application.

## Features

* Predicts the student's Math Score.
* Accepts student information through an interactive interface.
* Uses the same preprocessing steps used during model training.
* Provides a simple and user-friendly Streamlit interface.
* Uses a saved Machine Learning model for prediction.

## Input Features

The model uses the following features:

* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch
* Test Preparation Course
* Reading Score
* Writing Score

## Machine Learning

The project was developed using Python and Scikit-learn.

The trained model and preprocessing pipeline are saved using Joblib:

* `student_performance_model.pkl`
* `student_performance_preprocessor.pkl`

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

## Project Structure

```text
Student_Performance_Prediction/
│
├── app.py
├── student_performance_model.pkl
├── student_performance_preprocessor.pkl
├── requirements.txt
└── README.md
```

## How to Run the Project

First, install the required libraries:

```bash
pip install -r requirements.txt
```

Then run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

## How It Works

The user enters the student's information through the Streamlit interface.

The input data is passed through the saved preprocessing pipeline to transform it into the format expected by the Machine Learning model.

The trained model then predicts the student's Math Score and displays the prediction in the application.

## Goal

The goal of this project is to demonstrate how Machine Learning can be used to analyze student-related data and predict academic performance.
