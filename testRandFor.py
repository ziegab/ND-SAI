from sklearn.ensemble import RandomForestRegressor   # For regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Regression task
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

labels = []
inputs = []

with open(sys.argv[1]) as csvfile:
    reader = csv.reader(csvfile)
    n = 0
    for row in reader:
        labels.append(float(row[0]))
        tempinput = []
        for i in range(1, len(row)):
            tempinput.append(float(row[i]))
        inputs.append(tempinput)

numevents = len(labels)
print(numevents)

target_mean, target_std = np.mean(labels), np.std(labels)
labels_norm = (labels - target_mean) / target_std

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier (use RandomForestRegressor for regression tasks)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred)
r2_test = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f'Train R-squared: {r2_train:.4f}')
print(f'Test R-squared: {r2_test:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')

residuals = y_test - y_pred_test

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test, residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Random Forest Regression')
plt.show()