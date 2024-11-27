# Predictive-Modelling
import pandas as pd
# Load the dataset
data = pd.read_csv("Dataset.csv") # Replace with the actual file path
# Display first few rows of the dataset to understand its structure
print(data.head())
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
# Load the dataset
data = pd.read_csv("Dataset.csv") # Replace with the actual file path
# Convert categorical variables to numeric
data['Race'] = data['Race'].apply(lambda x: 1 if x == 'black' else 0)
data['Hisp'] = data['Hisp'].apply(lambda x: 1 if x == 'yes' else 0)
data['MaritalStatus'] = data['MaritalStatus'].apply(lambda x: 1 if x == 'yes' else 0)
# Convert 'Eduacation' column to numerical using Label Encoding
le = LabelEncoder()
data['Eduacation'] = le.fit_transform(data['Eduacation'])
# Select features and target variable
X = data[['Age', 'Race', 'Eduacation', 'Hisp', 'MaritalStatus', 'Earnings_1974', 'Earnings_1975']]
y = data['Earnings_1978']
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
# Output feature coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
