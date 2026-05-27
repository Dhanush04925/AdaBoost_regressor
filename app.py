import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("data/cardekho.csv")

# Remove Missing Values
data = data.dropna()

# Encode Categorical Columns
label_encoder = LabelEncoder()

for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Load Model
model = joblib.load("models/adaboost_model.pkl")

# Features and Target
target = "selling_price"

X = data.drop(target, axis=1)
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("🚗 CarDekho Price Prediction - AdaBoost Regressor")

# Model Performance
st.subheader("Model Performance")

st.success(f"Mean Absolute Error : {mae:.2f}")

st.success(f"R2 Score : {r2:.2f}")

# User Inputs
st.subheader("Enter Car Details")

input_data = {}

for column in X.columns:

    if X[column].dtype == "int64":
        value = st.number_input(
            f"{column}",
            value=int(X[column].mean())
        )

    else:
        value = st.number_input(
            f"{column}",
            value=float(X[column].mean())
        )

    input_data[column] = value

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict Price"):

    prediction = model.predict(input_df)

    st.success(f"Predicted Car Price : ₹ {prediction[0]:,.2f}")