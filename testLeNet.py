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

# LeNet CNN with the goal of regressing the boost of an AtoGG decay
# Importing information from csv file with each row = label, eta, phi, 15x15 flattened pixel image

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

def get_tensor_inputs_labels(arg):
    labels = []
    inputs = []
    with open(arg) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels.append(float(row[0]))
            tempinput = []
            for i in range(3, len(row)):
                tempinput.append(float(row[i]))
            inputs.append(tempinput)
    numevents = len(labels)
    # Pair elements together and shuffle
    combined = list(zip(labels, inputs))
    random.shuffle(combined)
    # Unzip back into separate lists
    labels, inputs = zip(*combined)
    # Convert back to lists (since zip() returns tuples)
    labels = list(labels)
    inputs = list(inputs)
    tensor_labels = torch.tensor(labels, dtype=torch.float32)
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    reshaped_tensor_inputs = [t.view(15,15) for t in tensor_inputs]
    # print(reshaped_tensor_inputs[0].shape)
    # target_mean, target_std = tensor_labels.mean(), tensor_labels.std()
    # tensor_labels_norm = (tensor_labels - target_mean) / target_std

    return tensor_labels, reshaped_tensor_inputs, numevents#, target_std, target_mean

datasets = []
numevents_tracker = []
# std_tracker = []
# mean_tracker = []
labels_tracker = []
inputs_tracker = []
argument_tracker = 0

for arg in sys.argv[1:]:
    argument_tracker += 1
    arg_labels, arg_inputs, arg_numevents = get_tensor_inputs_labels(arg)
    labels_tracker.append(arg_labels)
    inputs_tracker.append(arg_inputs)
    # datasets.append(CustomDataset(arg_inputs, arg_labels))
    numevents_tracker.append(arg_numevents)
    # std_tracker.append(arg_std)
    # mean_tracker.append(arg_mean)

# flattened = [x for sublist in arg_labels for x in sublist]
flattened = torch.cat(labels_tracker)
target_mean, target_std = flattened.mean(), flattened.std()
norm_labels_tracker = [(t-target_mean)/target_std for t in labels_tracker]
# print(argument_tracker)

for i in range(argument_tracker):
    datasets.append(CustomDataset(inputs_tracker[i], norm_labels_tracker[i]))

trains = []
vals = []

for i, dataset in enumerate(datasets):
    numevents = numevents_tracker[i]
    train, val = random_split(dataset, [int(numevents*0.7), numevents-int(numevents*0.7)])
    train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)
    trains.append(train_loader)
    vals.append(val_loader)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 15x15 -> 15x15
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 15x15 -> 7x7
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 7x7 -> 3x3
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 3x3 -> 1x1
        self.fc1 = nn.Linear(16 * 1 * 1, 120)  # Fully connected
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 1)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.pool1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.pool2(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)  # Flatten for FC layers
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)  # No activation (logits)
        return x
    
# Training model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = LeNet5()

# Define Loss and Optimizer
criterion = nn.MSELoss()  # Regression loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)  # Ensure biases start at zero

model.apply(init_weights)

# Training Loop
nb_epochs = 50
traininglosses = []
validationlosses = []
for epoch in range(nb_epochs):
    model.train()
    # running_loss = 0.0
    losses = list()
    for images, labels in chain(*trains):  # Use your dataset loader
        images, labels = images, labels
        # print(images.unsqueeze(1).shape)

        optimizer.zero_grad()
        outputs = model(images.unsqueeze(1))
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
    for images, labels in chain(*vals): 
        # 1. forward
        with torch.no_grad():
            l = model(images.unsqueeze(1)) 

        #2. compute the objective function
        loss = criterion(l, labels.view(-1, 1)) 

        losses.append(loss.item())

    print(f'Epoch {epoch +1}, validation loss: {torch.tensor(losses).mean():.2f}')
    validationlosses.append(torch.tensor(losses).mean())

model.eval()
x = []
y = []
with torch.no_grad():
    for i,loader in enumerate(trains):
        for test_images, test_labels_norm in loader:
            test_labels_norm = test_labels_norm.unsqueeze(1)
            outputs_norm = model(test_images.unsqueeze(1))
            outputs = outputs_norm * target_std + target_mean
            test_labels = test_labels_norm * target_std + target_mean
            # print("Target range:", torch.min(test_labels), torch.max(test_labels))
            # print("Prediction range:", torch.min(outputs), torch.max(outputs))
            for j in range(32):
                if outputs[j].item() > 0:
                    x.append(test_labels[j].item())
                    y.append(outputs[j].item())


# # predicted - true /true
plt.scatter(x, y, c='blue', alpha=0.5)
plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
plt.title('True vs. Predicted Boost for LeNet CNN')
plt.savefig(f"LatestPredTruePlot.pdf")


print(f"Printing Epochs={nb_epochs} plots.")
# Training Losses plot
nb_epochslist = list(range(0, nb_epochs))
last_trainloss = traininglosses[-1]

plt.figure(1)
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
plt.figure(2)
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