# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n==============================")
print("House Price Prediction System")
print("==============================")

# -------------------------------
# Take User Inputs First
# -------------------------------
gr_liv_area = float(input("Enter Living Area (in square feet): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathroom = int(input("Enter Number of Bathrooms: "))

# -------------------------------
# Load Training Dataset
# -------------------------------
train_data = pd.read_csv("train.csv")

train_data = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Rename column
train_data.rename(columns={'FullBath': 'Bathroom'}, inplace=True)

# -------------------------------
# Define Features and Target
# -------------------------------
X = train_data[['GrLivArea', 'BedroomAbvGr', 'Bathroom']]
y = train_data['SalePrice']

# -------------------------------
# Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nModel Performance")
print("---------------------")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -------------------------------
# Predict User House Price
# -------------------------------
user_data = pd.DataFrame({
    'GrLivArea': [gr_liv_area],
    'BedroomAbvGr': [bedrooms],
    'Bathroom': [bathroom]
})

predicted_price = model.predict(user_data)

print("\nEstimated House Price: $", round(predicted_price[0], 2))

# -------------------------------
# Create Submission File
# -------------------------------
test_data = pd.read_csv("test.csv")

test_data = test_data[['Id', 'GrLivArea', 'BedroomAbvGr', 'FullBath']]

test_data.rename(columns={'FullBath': 'Bathroom'}, inplace=True)

X_final = test_data[['GrLivArea', 'BedroomAbvGr', 'Bathroom']]

predictions = model.predict(X_final)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": predictions
})

submission.to_csv("final_submission.csv", index=False)

print("\nSubmission file 'final_submission.csv' created successfully!")