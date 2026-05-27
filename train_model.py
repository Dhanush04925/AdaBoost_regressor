import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Load Dataset
data = pd.read_csv("data/cardekho.csv")

# Drop Missing Values
data = data.dropna()

# Encode Categorical Columns
label_encoder = LabelEncoder()

for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Target Column
target = "selling_price"

# Features and Target
X = data.drop(target, axis=1)
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create AdaBoost Regressor
base_model = DecisionTreeRegressor(max_depth=4)

model = AdaBoostRegressor(
    estimator=base_model,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "models/adaboost_model.pkl")

print("Model trained and saved successfully!")