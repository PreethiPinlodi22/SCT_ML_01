
# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
train_data = pd.read_csv("train.csv")

# Select required columns
train_data = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']]

# Feature Engineering
train_data['TotalBath'] = train_data['FullBath'] + (0.5 * train_data['HalfBath'])

# Define features and target
X = train_data[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
y = train_data['SalePrice']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Save submission file
test_data = pd.read_csv("test.csv")
test_data['TotalBath'] = test_data['FullBath'] + (0.5 * test_data['HalfBath'])
X_final = test_data[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
predictions = model.predict(X_final)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": predictions
})

submission.to_csv("final_submission.csv", index=False)
print("Submission file created successfully!")
