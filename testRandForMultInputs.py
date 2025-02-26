import numpy as np
import csv
import sys
from sklearn.ensemble import RandomForestRegressor   # For regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Regression task
from sklearn.datasets import make_regression
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
    # target_mean, target_std = np.mean(labels), np.std(labels)
    # labels_norm = (labels - target_mean) / target_std
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test#, target_mean, target_std

input_trains = []
input_vals = []
label_trains = []
label_vals = []
# std_tracker = []
# mean_tracker = []


for arg in sys.argv[1:]:
    # X_train, X_test, y_train, y_test, target_mean, target_std = get_labels_inputs(arg)
    X_train, X_test, y_train, y_test = get_labels_inputs(arg)
    input_trains.append(X_train)
    input_vals.append(X_test)
    label_trains.append(y_train)
    label_vals.append(y_test)
    # std_tracker.append(target_std)
    # mean_tracker.append(target_mean)

input_train = list(chain(*input_trains))
input_val = list(chain(*input_vals))
label_train = list(chain(*label_trains))
label_val = list(chain(*label_vals))

alllabels = label_train + label_val

target_mean, target_std = np.mean(alllabels), np.std(alllabels)
label_train_norm = (label_train - target_mean) / target_std
label_val_norm = (label_val - target_mean) / target_std

# Initialize Random Forest Classifier (use RandomForestRegressor for regression tasks)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(input_train, label_train_norm)

# Make predictions on the test set
label_pred = model.predict(input_train)
label_pred_val = model.predict(input_val)

# r2_train = r2_score(y_train, y_pred)
# r2_test = r2_score(y_test, y_pred_test)
mse = mean_squared_error(label_train_norm, label_pred)
msev = mean_squared_error(label_val_norm, label_pred_val)
# print(f'Train R-squared: {r2_train:.4f}')
# print(f'Test R-squared: {r2_test:.4f}')
print(f'Training Mean Squared Error (MSE): {mse:.4f}')
print(f'Validation Mean Squared Error (MSE): {msev:.4f}')

x = [] # true
y = [] # predicted
# for i,input in enumerate(input_vals):
#     predictions_norm = model.predict(input)
#     predictions = [(x * std_tracker[i] + mean_tracker[i]) for x in predictions_norm]
#     labels_norm = label_vals[i]
#     labels = [(x * std_tracker[i] + mean_tracker[i]) for x in labels_norm]
#     for j in range(len(input)):
#         # if 0 < predictions[j] < 1500:
#             x.append(labels[j])
#             y.append(predictions[j])
# for i in range(len(input_val)):
predictions_norm = model.predict(input_train)
predictions = [(x * target_std + target_mean) for x in predictions_norm]
labels = [(x * target_std + target_mean) for x in label_train_norm]
x.append(labels)
y.append(predictions)

# # predicted - true /true
plt.scatter(x, y, c='blue', alpha=0.5)
plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
plt.title('True vs. Predicted Boost for MLP Regressor')
plt.savefig(f"LatestPredTruePlot.pdf")