import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
import csv
import sys
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import chain
import glob
from pprint import pprint
# from sklearn.linear_model import LinearRegression

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    
class CustomDatasetFeatures(Dataset):
    def __init__(self, data, features, labels, transform=None):
        self.data = data
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        feature = self.features[idx]
        y = self.labels[idx]
        return x, feature, y
    
def split_train_val(arg):
    labels = []
    inputs = []
    etas = []
    with open(arg) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(f"Row: {len(row)}")
            if min_boost < float(row[0]) < max_boost:
                labels.append(float(row[0]))
                etas.append((float(row[2])*6)-3)
                tempinput = []
                for i in range(4, len(row)):
                    tempinput.append(float(row[i]))
                # print(f"Extracted input: {len(tempinput)}")
                inputs.append(tempinput)
    numevents = len(labels)
    # Pair elements together and shuffle
    extra_features_list = etas
    combined = list(zip(labels, inputs, extra_features_list))
    random.shuffle(combined)
    split_idx = int(len(combined) * 0.7)
    # print(split_idx)
    train_combined = combined[:split_idx]
    val_combined = combined[split_idx:]
    # print(f"Length of train_combined: {len(train_combined)}")
    # print(f"Length of val_combined: {len(val_combined)}")
    # Unzip back into separate lists
    train_labels, train_inputs, train_extra_features = zip(*train_combined)
    val_labels, val_inputs, val_extra_features = zip(*val_combined)
    # Convert back to lists (since zip() returns tuples)
    train_labels = list(train_labels)
    train_inputs = list(train_inputs)
    train_extra_features = list(train_extra_features)
    val_labels = list(val_labels)
    val_inputs = list(val_inputs)
    val_extra_features = list(val_extra_features)
    # print(len(train_inputs), len(train_labels))
    tensor_train_inputs = torch.stack([torch.tensor(t, dtype=torch.float32).view(1,input_size,input_size) for t in train_inputs])
    tensor_val_inputs = torch.stack([torch.tensor(t, dtype=torch.float32).view(1,input_size,input_size) for t in val_inputs])
    return train_labels, val_labels, tensor_train_inputs, tensor_val_inputs, train_extra_features, val_extra_features

def get_tensor_inputs_labels(arg):
    labels = []
    inputs = []
    eta = []
    with open(arg) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if min_boost < float(row[0]) < max_boost:
                labels.append(float(row[0]))
                eta.append((float(row[2])*6)-3)
                tempinput = []
                for i in range(4, len(row)):
                    tempinput.append(float(row[i]))
                inputs.append(tempinput)
    numevents = len(labels)
    # Pair elements together and shuffle
    etas = eta
    combined = list(zip(labels, inputs, etas))
    random.shuffle(combined)
    # Unzip back into separate lists
    labels, inputs, etas = zip(*combined)
    # Convert back to lists (since zip() returns tuples)
    labels = list(labels)
    inputs = list(inputs)
    etas = list(etas)
    tensor_labels = torch.tensor(labels, dtype=torch.float32)
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    tensor_etas = torch.tensor(etas, dtype=torch.float32)
    reshaped_tensor_inputs = [t.view(input_size,input_size) for t in tensor_inputs]
    # print(reshaped_tensor_inputs[0].shape)
    # target_mean, target_std = tensor_labels.mean(), tensor_labels.std()
    # tensor_labels_norm = (tensor_labels - target_mean) / target_std

    return tensor_labels, reshaped_tensor_inputs, numevents, tensor_etas#, target_std, target_mean

datasets = []
numevents_tracker = []
# std_tracker = []
# mean_tracker = []
labels_tracker = []
inputs_tracker = []
etas_tracker = []
# argument_tracker = 0
totaleventcounter = 0

labels_train_tracker = []
labels_val_tracker = []
inputs_train_tracker = []
inputs_val_tracker = []
features_train_tracker = []
features_val_tracker = []
# etas_tracker = []
argument_tracker = 0

file_dir = str(sys.argv[1])
print(file_dir)
csv_files = glob.glob(f"{file_dir}/*.csv")

input_size = 32
max_boost = 210
min_boost = 20
batchsize = 64

for arg in (csv_files):
# for arg in sys.argv[2:]:
    argument_tracker += 1
    arg_labels, arg_inputs, arg_numevents, arg_etas = get_tensor_inputs_labels(arg)
    totaleventcounter += arg_numevents
    labels_tracker.append(arg_labels)
    inputs_tracker.append(arg_inputs)
    etas_tracker.append(arg_etas)
    # datasets.append(CustomDataset(arg_inputs, arg_labels))
    numevents_tracker.append(arg_numevents)
    arg_train_labels, arg_val_labels, arg_train_inputs, arg_val_inputs, arg_train_features, arg_val_features = split_train_val(arg)
    labels_train_tracker.append(arg_train_labels)
    labels_val_tracker.append(arg_val_labels)
    inputs_train_tracker.append(arg_train_inputs)
    inputs_val_tracker.append(arg_val_inputs)
    features_train_tracker.append(arg_train_features)
    features_val_tracker.append(arg_val_features)
    # if argument_tracker == 1:
    #     break

# pprint(labels_train_tracker)
tensor_train_inputs = [tensor for sublist in inputs_train_tracker for tensor in sublist]
tensor_val_inputs = [tensor for sublist in inputs_val_tracker for tensor in sublist]
tensor_train_features = [torch.tensor([inner]) for outer in features_train_tracker for inner in outer]
tensor_val_features = [torch.tensor([inner]) for outer in features_val_tracker for inner in outer]
tensor_train_labels = [torch.tensor([inner]) for outer in labels_train_tracker for inner in outer]
tensor_val_labels = [torch.tensor([inner]) for outer in labels_val_tracker for inner in outer]

# print(len(tensor_train_inputs), len(tensor_train_labels))

flattened = torch.cat(tensor_train_labels)
print(f"Length of flattened train_labels: {len(flattened)}")

# target_mean, target_std = flattened.mean(), flattened.std()
# norm_labels_train_tracker = [(t-target_mean)/target_std for t in tensor_train_labels]
# # pprint(norm_labels_train_tracker)
# norm_labels_val_tracker = [(t-target_mean)/target_std for t in tensor_val_labels]
# print(target_mean, target_std)
# norm_labels_tracker = [(t-target_mean)/target_std for t in labels_tracker]

norm_labels_train_tracker = [torch.log(t+1) for t in tensor_train_labels]
# pprint(norm_labels_train_tracker)
norm_labels_val_tracker = [torch.log(t+1) for t in tensor_val_labels]
# print(target_mean, target_std)
norm_labels_tracker = [torch.log(t+1) for t in labels_tracker]

flattened_features = torch.cat(tensor_train_features)
print(f"Length of flattened train_features: {len(flattened_features)}")


target_mean_features, target_std_features = flattened_features.mean(), flattened_features.std()
norm_features_train_tracker = [(t-target_mean_features)/target_std_features for t in tensor_train_features]
# pprint(norm_labels_train_tracker)
norm_features_val_tracker = [(t-target_mean_features)/target_std_features for t in tensor_val_features]
print(target_mean_features, target_std_features)
norm_etas_tracker = [(t-target_mean_features)/target_std_features for t in etas_tracker]


class MinMaxNormalize:
    def __init__(self):
        pass
    
    def __call__(self, image):
        # Convert the image to a tensor
        image_tensor = transforms.ToTensor()(image)
        
        # Normalize the image using min-max normalization
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        
        # Apply min-max normalization to scale to [0, 1]
        normalized_image = (image_tensor - min_val) / (max_val - min_val)
        
        return normalized_image


transform = transforms.Compose([
    MinMaxNormalize()
])

# for i in range(argument_tracker):
#     datasets.append(CustomDataset(inputs_tracker[i], norm_labels_tracker[i], transform=transform))

# inputs_train_flat = torch.cat([torch.cat(inputs) for inputs in inputs_train_tracker])
# # norm_labels_train_flat = torch.cat([torch.cat(labels) for labels in norm_labels_train_tracker])
# inputs_val_flat = torch.cat([torch.cat(inputs) for inputs in inputs_val_tracker])
# # norm_labels_val_flat = torch.cat([torch.cat(labels) for labels in norm_labels_val_tracker])

for i in range(argument_tracker):
    datasets.append(CustomDatasetFeatures(inputs_tracker[i], etas_tracker[i], norm_labels_tracker[i]))#, transform=transform))

datasets_validate = []
predictions_mean = []
predictions_std = []

for arg in sys.argv[2:]:
# for arg in (csv_files):
    arg_labels, arg_inputs, arg_numevents, arg_etas = get_tensor_inputs_labels(arg)
    datasets_validate.append(CustomDatasetFeatures(arg_inputs,arg_etas, arg_labels))
    predictions_mean.append(arg_labels.mean())
    predictions_std.append(arg_labels.std())

trains = []
vals = []
predictions = []

for i, dataset in enumerate(datasets):
    numevents = numevents_tracker[i]
    train, val = random_split(dataset, [int(numevents*0.7), numevents-int(numevents*0.7)])
    train_loader_group = DataLoader(train, batch_size=batchsize, shuffle=True, drop_last=True)
    val_loader_group = DataLoader(val, batch_size=batchsize, shuffle=False, drop_last=True)
    trains.append(train_loader_group)
    vals.append(val_loader_group)

for dataset in datasets_validate:
    prediction_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    predictions.append(prediction_loader)

combined_tensors = list(zip(norm_labels_train_tracker, tensor_train_inputs, norm_features_train_tracker))
random.shuffle(combined_tensors)
norm_labels_train_tracker, tensor_train_inputs, norm_features_train_tracker = map(list, zip(*combined_tensors))
combined_val_tensors = list(zip(norm_labels_val_tracker, tensor_val_inputs, norm_features_val_tracker))
random.shuffle(combined_val_tensors)
norm_labels_val_tracker, tensor_val_inputs, norm_features_val_tracker = map(list, zip(*combined_val_tensors))

train_dataset = CustomDatasetFeatures(torch.cat(tensor_train_inputs), torch.cat(norm_features_train_tracker), torch.cat(norm_labels_train_tracker))#, transform=transform)
val_dataset = CustomDatasetFeatures(torch.cat(tensor_val_inputs), torch.cat(norm_features_val_tracker), torch.cat(norm_labels_val_tracker))#, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, drop_last=True)

class LeNet5(nn.Module):
    def __init__(self, extra_feature_dim=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 32x32 -> 32x32
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 16x16 -> 12x12
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 12x12 -> 6x6
        self.fc1 = nn.Linear((16 * 6 * 6) + extra_feature_dim, 784)  # Fully connected
        self.bn3 = nn.BatchNorm1d(784)
        self.fc2 = nn.Linear(784, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 1) 

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, extra_feature):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        # x = self.pool1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        # x = self.pool2(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)  # Flatten for FC layers
        extra_feature = extra_feature.view(x.size(0), -1)
        x = torch.cat((x, extra_feature), dim=1)

        x = F.leaky_relu(self.bn3(self.fc1(x)))
        x = F.leaky_relu(self.bn4(self.fc2(x)))
        x = F.leaky_relu(self.bn5(self.fc3(x)))  # No activation (logits)
        x = self.fc4(x)

        # x = F.relu((self.conv1(x)))
        # # x = self.pool1(x)
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu((self.conv2(x)))
        # # x = self.pool2(x)
        # x = F.max_pool2d(x, 2, 2)
        # x = torch.flatten(x, 1)  # Flatten for FC layers
        # x = F.relu((self.fc1(x)))
        # x = F.relu((self.fc2(x)))
        # x = self.fc3(x)  # No activation (logits)

        # x = F.leaky_relu((self.fc1(x)))
        # x = F.leaky_relu((self.fc2(x)))
        # x = F.leaky_relu(self.fc3(x))  # No activation (logits)
        # x = self.fc4(x)
        return x
    
# Training model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = LeNet5(extra_feature_dim=1)

class LogMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log1p((y_pred - y_true) ** 2))
    
class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

# Define Loss and Optimizer
criterion = nn.MSELoss()  # Regression loss function
# criterion = LogCoshLoss()
# criterion = nn.SmoothL1Loss(beta=10.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Ensure biases start at zero

model.apply(init_weights)

# Training Loop
nb_epochs = 100
traininglosses = []
validationlosses = []
for epoch in range(nb_epochs):
    model.train()
    # running_loss = 0.0
    losses = list()
    for images, extra_features, labels in train_loader:
            images, extra_features, labels = images, extra_features, labels
            # print(f'Image batch size: {images.size()}')
            # print(f'Label batch size: {labels.size()}')
            # print(images.unsqueeze(1).shape)

            optimizer.zero_grad()
            outputs = model(images.unsqueeze(1), extra_features.unsqueeze(1))
            loss = criterion(outputs, labels.view(-1, 1))  # Ensure proper shape
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # running_loss += loss.item()
            losses.append(loss.item())

    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print(f'Epoch {epoch +1}, training loss: {torch.tensor(losses).mean():.2f}')
    traininglosses.append(torch.tensor(losses).mean())

    losses = list()
    # for loader in vals:
    # for images, labels in chain(*vals): 
    for images, extra_features, labels in val_loader:
            # 1. forward
            with torch.no_grad():
                l = model(images.unsqueeze(1), extra_features.unsqueeze(1)) 

            #2. compute the objective function
            loss = criterion(l, labels.view(-1, 1)) 

            losses.append(loss.item())

    print(f'Epoch {epoch +1}, validation loss: {torch.tensor(losses).mean():.2f}')
    validationlosses.append(torch.tensor(losses).mean())

model.eval()
# # all labels together
# x_train = []
# y_train = []
# with torch.no_grad():
#     # for i,loader in enumerate(trains):
#         for test_images, test_labels_norm in train_loader:
#             test_labels_norm = test_labels_norm.unsqueeze(1)
#             outputs_norm = model(test_images.unsqueeze(1))
#             outputs = (outputs_norm * target_std) + target_mean
#             test_labels = (test_labels_norm * target_std) + target_mean
#             # print("Target range:", torch.min(test_labels), torch.max(test_labels))
#             # print("Prediction range:", torch.min(outputs), torch.max(outputs))
#             for j in range(32):
#                 # if 0 < outputs[j].item() < 1500:
#                     x_train.append(test_labels[j].item())
#                     y_train.append(outputs[j].item())
#         # if i == 0:
#         #     break

# x_val = []
# y_val = []
# with torch.no_grad():
#     # for i,loader in enumerate(vals):
#         for test_images, test_labels_norm in val_loader:
#             test_labels_norm = test_labels_norm.unsqueeze(1)
#             outputs_norm = model(test_images.unsqueeze(1))
#             outputs = (outputs_norm * target_std) + target_mean
#             test_labels = (test_labels_norm * target_std) + target_mean
#             # print("Target range:", torch.min(test_labels), torch.max(test_labels))
#             # print("Prediction range:", torch.min(outputs), torch.max(outputs))
#             for j in range(32):
#                 # if 0 < outputs[j].item() < 1500:
#                     x_val.append(test_labels[j].item())
#                     y_val.append(outputs[j].item())
#         # if i == 0:
#         #     break

# labels grouped by mass point
x_train = []
y_train = []
with torch.no_grad():
    for i,loader in enumerate(trains):
        x_train_temp = []
        y_train_temp = []
        for test_images, test_features, test_labels_norm in loader:
            test_labels_norm = test_labels_norm.unsqueeze(1)
            outputs_norm = model(test_images.unsqueeze(1), test_features.unsqueeze(1))
            # outputs = (outputs_norm * target_std) + target_mean
            # test_labels = (test_labels_norm * target_std) + target_mean
            outputs = torch.exp(outputs_norm)-1
            test_labels = torch.exp(test_labels_norm)-1
            # print("Target range:", torch.min(test_labels), torch.max(test_labels))
            # print("Prediction range:", torch.min(outputs), torch.max(outputs))
            for j in range(batchsize):
                # if 0 < outputs[j].item() < 1500:
                    x_train_temp.append(test_labels[j].item())
                    y_train_temp.append(outputs[j].item())
        x_train.append(x_train_temp)
        y_train.append(y_train_temp)
        # if i == 0:
        #     break

x_val = []
y_val = []
with torch.no_grad():
    for i,loader in enumerate(vals):
        x_val_temp = []
        y_val_temp = []
        for test_images, test_features, test_labels_norm in loader:
            test_labels_norm = test_labels_norm.unsqueeze(1)
            outputs_norm = model(test_images.unsqueeze(1), test_features.unsqueeze(1))
            # outputs = (outputs_norm * target_std) + target_mean
            # test_labels = (test_labels_norm * target_std) + target_mean
            outputs = torch.exp(outputs_norm)-1
            test_labels = torch.exp(test_labels_norm)-1
            # print("Target range:", torch.min(test_labels), torch.max(test_labels))
            # print("Prediction range:", torch.min(outputs), torch.max(outputs))
            for j in range(batchsize):
                # if 0 < outputs[j].item() < 1500:
                    x_val_temp.append(test_labels[j].item())
                    y_val_temp.append(outputs[j].item())
        x_val.append(x_val_temp)
        y_val.append(y_val_temp)
        # if i == 0:
        #     break

x_pred = []
y_pred = []
with torch.no_grad():
    for i,loader in enumerate(predictions):
        x_pred_temp = []
        y_pred_temp = []
        for test_images, test_features, test_labels_norm in loader:
            test_labels_norm = test_labels_norm.unsqueeze(1)
            outputs_norm = model(test_images.unsqueeze(1), test_features.unsqueeze(1))
            # outputs = ((outputs_norm * target_std) + target_mean) / 2.727
            outputs = torch.exp(outputs_norm)-1
            # test_labels = torch.exp(test_labels_norm)-1
            # outputs = (outputs_norm * predictions_std[i]) + predictions_mean[i]

            true = test_labels_norm.numpy().reshape(-1, 1)
            pred = outputs.numpy().reshape(-1, 1)

            # reg = LinearRegression().fit(pred, true)
            # corrected_pred = reg.predict(pred)

            # test_labels = (test_labels_norm * target_std) + target_mean
            test_labels = torch.exp(test_labels_norm)-1
            # print("Target range:", torch.min(test_labels), torch.max(test_labels))
            # print("Prediction range:", torch.min(outputs), torch.max(outputs))
            for j in range(batchsize):
                # if 0 < outputs[j].item() < 1500:
                if 50 < test_labels_norm[j] < max_boost:
                    x_pred_temp.append(test_labels_norm[j].item())
                    y_pred_temp.append(outputs[j].item())
        x_pred.append(x_pred_temp)
        y_pred.append(y_pred_temp)

# # all labels together
# # # predicted - true /true
# plt.figure(1)
# plt.scatter(x_train, y_train, c='blue', alpha=0.6, s=1)
# plt.plot([0.05, 0.25], [0.05, 0.25], color='black', linestyle='-', linewidth=1, label='Predicted = True')
# # plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
# plt.xlabel('True Boost')
# plt.ylabel('Predicted Boost')
# plt.title('True vs. Predicted Boost for LeNet CNN (Training Data)')
# plt.savefig(f"TrainPredTrueLeNetMass.pdf")

# plt.figure(2)
# plt.scatter(x_val, y_val, c='blue', alpha=0.6, s=1)
# plt.plot([0.05, 0.25], [0.05, 0.25], color='black', linestyle='-', linewidth=1, label='Predicted = True')
# # plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
# plt.xlabel('True Boost')
# plt.ylabel('Predicted Boost')
# plt.title('True vs. Predicted Boost for LeNet CNN (Validation Data)')
# plt.savefig(f"ValPredTrueLeNetMass.pdf")

print(len(x_train), len(x_val))

# labels grouped by mass point
colors_train = plt.cm.viridis(np.linspace(0, 1, len(x_train)))
colors_val = plt.cm.viridis(np.linspace(0, 1, len(x_val)))
colors_pred = plt.cm.viridis(np.linspace(0, 1, len(x_pred)))
plt.figure(1)
for i in range(len(x_train)):
    plt.scatter(x_train[i], y_train[i], color=colors_train[i], label=f'Mass Point {i+1}', alpha=0.6, s=1)
# plt.plot([0.05, 0.25], [0.05, 0.25], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.plot([0, max_boost], [0, max_boost], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
# plt.legend()
plt.title('True vs. Predicted Boost for LeNet CNN (Training Data)')
plt.savefig(f"TrainPredTrueLeNetMass.pdf")

plt.figure(2)
for i in range(len(x_val)):
    plt.scatter(x_val[i], y_val[i], color=colors_val[i], label=f'Mass Point {i+1}', alpha=0.6, s=1)
# plt.plot([0.05, 0.25], [0.05, 0.25], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.plot([0, max_boost], [0, max_boost], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
# plt.legend()
plt.title('True vs. Predicted Boost for LeNet CNN (Validation Data)')
plt.savefig(f"ValPredTrueLeNetMass.pdf")

plt.figure(6)
for i in range(len(x_pred)):
    plt.scatter(x_pred[i], y_pred[i], color=colors_pred[i], label=f'Mass Point {i+1}', alpha=0.6, s=1)
# plt.plot([0.05, 0.25], [0.05, 0.25], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.plot([0, max_boost], [0, max_boost], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
# plt.legend()
plt.title('True vs. Predicted Boost for LeNet CNN (Training Data)')
plt.savefig(f"PredPredTrueLeNetMass.pdf")

# flattened_inputs_tracker = [item for sublist in inputs_tracker for item in sublist]
# inputs_array = np.array(flattened_inputs_tracker, dtype=np.float32)
# inputs_tracker_tensor = torch.tensor(inputs_array).unsqueeze(1)
# print(inputs_tracker_tensor.shape)
# norm_predictions_tracker = model(inputs_tracker_tensor)
# predictions = (norm_predictions_tracker * target_std) + target_mean

# flattened_labels_tracker = [item for sublist in labels_tracker for item in sublist]
# print(len(flattened_labels_tracker))
# # norm_predictions_list = norm_predictions_tracker.tolist()
# hist_weight_train = []
# for i in range(len(flattened_labels_tracker)):
#     true = flattened_labels_tracker[i].item()
#     hist_weight_train.append((predictions[i].item() - true) / true)

# plt.figure(5)
# x_range = (100, max(flattened_labels_tracker))
# binset = [np.linspace(x_range[0], x_range[1], 30), 30]
# plt.hist2d(flattened_labels_tracker, [item for sublist in etas_tracker for item in sublist], bins=binset, weights=hist_weight_train, cmap='plasma')
# plt.colorbar(label='(Pred-True)/True')
# plt.xlabel('Boost')
# plt.ylabel('Eta')
# plt.title('2D Histogram')
# plt.savefig(f"Test2DHist.pdf")

print(f"Printing Epochs={nb_epochs} plots.")
# Training Losses plot
nb_epochslist = list(range(0, nb_epochs))
last_trainloss = traininglosses[-1]

plt.figure(3)
plt.scatter(nb_epochslist, traininglosses, c="orange", alpha=0.5)
plt.plot([-1, nb_epochs+1], [last_trainloss, last_trainloss], 'p--', label=f'Final Training Loss = {last_trainloss:.2f}', alpha=0.2)
plt.xlim(0, nb_epochs)
plt.ylim(0, 1)
# plt.plot([0,nb_epochs], [0, 1], color='black', linestyle='-', linewidth=1, label='Training Losses ')
plt.xlabel('Epochs')
plt.ylabel('Training Losses')
plt.title(f"LeNet CNN")
plt.legend()
plt.savefig(f"LeNetepoch{nb_epochs}v1.pdf", format="pdf")

# Validation Losses plot

last_valloss = validationlosses[-1]
plt.figure(4)
plt.scatter(nb_epochslist, validationlosses, c="pink", alpha=0.5)
plt.plot([-1, nb_epochs+1], [last_valloss, last_valloss], 'p--', label=f'Final Validation Loss = {last_valloss:.2f}', alpha=0.2)
plt.xlim(0, nb_epochs)
plt.ylim(0, 1)
# plt.plot([0,nb_epochs], [0, 1], color='black', linestyle='-', linewidth=1, label='Training Losses ')
plt.xlabel('Epochs')
plt.ylabel('Validation Losses')
plt.title(f"LeNet CNN")
plt.legend()
plt.savefig(f"ValidationLeNetepoch{nb_epochs}v1.pdf", format="pdf")