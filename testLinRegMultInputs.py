import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import chain

def get_labels_inputs(arg):
    labels = []
    inputs = []
    with open(arg) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels.append(float(row[0]))
            tempinput = []
            for i in range(1, len(row)):
                tempinput.append(float(row[i]))
            inputs.append(tempinput)
    numevents = len(labels)
    target_mean, target_std = np.mean(labels), np.std(labels)
    labels_norm = (labels - target_mean) / target_std
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels_norm, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, target_mean, target_std

input_trains = []
input_vals = []
label_trains = []
label_vals = []
std_tracker = []
mean_tracker = []


for arg in sys.argv[1:]:
    X_train, X_test, y_train, y_test, target_mean, target_std = get_labels_inputs(arg)
    input_trains.append(X_train)
    input_vals.append(X_test)
    label_trains.append(y_train)
    label_vals.append(y_test)
    std_tracker.append(target_std)
    mean_tracker.append(target_mean)

## Linear Regression
input_train = list(chain(*input_trains))
input_val = list(chain(*input_vals))
label_train = list(chain(*label_trains))
label_val = list(chain(*label_vals))
reg = LinearRegression().fit(input_train, label_train)
r2_train = reg.score(input_train, label_train)
r2_test = reg.score(input_val, label_val)
print(f'R-squared (Train): {r2_train:.4f}')
print(f'R-squared (Test): {r2_test:.4f}')

mse = mean_squared_error(label_train, reg.predict(input_train))
print(f'Training Mean Squared Error: {mse:.4f}')
mse = mean_squared_error(label_val, reg.predict(input_val))
print(f'Validation Mean Squared Error: {mse:.4f}')

x = [] # true
y = [] # predicted
for i,input in enumerate(input_trains):
    predictions_norm = reg.predict(input)
    predictions = [(x * std_tracker[i] + mean_tracker[i]) for x in predictions_norm]
    labels_norm = label_trains[i]
    labels = [(x * std_tracker[i] + mean_tracker[i]) for x in labels_norm]
    for j in range(len(input)):
        if 0 < predictions[j] < 1500:
            x.append(labels[j])
            y.append(predictions[j])

# # predicted - true /true
plt.scatter(x, y, c='blue', alpha=0.5)
plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
plt.title('True vs. Predicted Boost for MLP Regressor')
plt.savefig(f"LatestPredTruePlot.pdf")