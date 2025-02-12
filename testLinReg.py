import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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


## Linear Regression
# reg = LinearRegression().fit(inputs, labels_norm)
# print(reg.score(inputs, labels_norm))
reg = LinearRegression().fit(X_train, y_train)
r2_train = reg.score(X_train, y_train)
r2_test = reg.score(X_test, y_test)
print(f'R-squared (Train): {r2_train:.4f}')
print(f'R-squared (Test): {r2_test:.4f}')
# print(reg.coef_)
# print(reg.intercept_)
# print(reg.feature_names_in_)

mse = mean_squared_error(y_test, reg.predict(X_test))
print(f'Mean Squared Error: {mse:.4f}')

y_pred = reg.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()