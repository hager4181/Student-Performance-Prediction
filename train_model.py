import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("exams.csv")


# Features and target
X = df.drop("math score", axis=1)
y = df["math score"]


# Feature types
categorical_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

numerical_features = [
    "reading score",
    "writing score"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Train model
model = LinearRegression()
model.fit(X_train_processed, y_train)


# Prediction
y_pred = model.predict(X_test_processed)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Save model and preprocessor
joblib.dump(model, "student_performance_model.pkl")
joblib.dump(preprocessor, "student_performance_preprocessor.pkl")

print("Model and preprocessor saved successfully!")