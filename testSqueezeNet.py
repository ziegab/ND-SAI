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

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.bn_squeeze = nn.BatchNorm2d(squeeze_channels)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.bn_expand1x1 = nn.BatchNorm2d(expand_channels)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.bn_expand3x3 = nn.BatchNorm2d(expand_channels)

    def forward(self, x):
        x = F.relu(self.bn_squeeze(self.squeeze(x)))
        return torch.cat([F.relu(self.bn_expand1x1(self.expand1x1(x))), F.relu(self.bn_expand3x3(self.expand3x3(x)))], dim=1)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fire2 = FireModule(96, 16, 64)
        self.fire3 = FireModule(128, 16, 64)
        self.fire4 = FireModule(128, 32, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fire5 = FireModule(256, 32, 128)
        self.fire6 = FireModule(256, 48, 192)
        self.fire7 = FireModule(384, 48, 192)
        self.fire8 = FireModule(384, 64, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fire9 = FireModule(512, 64, 256)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.pool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.pool3(x)
        x = self.fire9(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_avg_pool(x)
        return torch.flatten(x, 1)

# Instantiate the model
model = SqueezeNet(num_classes=1)
# print(model)

# Define loss & optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)  # Ensure biases start at zero

model.apply(init_weights)

# Training loop
nb_epochs = 50
traininglosses = []
validationlosses = []
for epoch in range(nb_epochs):
    model.train()
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
    for i,train_loader in enumerate(vals):
        for test_images, test_labels_norm in val_loader:
            test_labels_norm = test_labels_norm.unsqueeze(1)
            outputs_norm = model(test_images.unsqueeze(1))
            outputs = outputs_norm * target_std + target_mean
            test_labels = test_labels_norm * target_std + target_mean
            # print("Target range:", torch.min(test_labels), torch.max(test_labels))
            # print("Prediction range:", torch.min(outputs), torch.max(outputs))
            for j in range(32):
                # if outputs[j].item() > 0:
                    x.append(test_labels[j].item())
                    y.append(outputs[j].item())


# # predicted - true /true
plt.scatter(x, y, c='blue', alpha=0.5)
plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
plt.title('True vs. Predicted Boost for SqueezeNet CNN')
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
plt.title(f"SqueezeNet CNN")
plt.legend()
plt.savefig(f"SqueezeNetepoch{nb_epochs}v1.pdf", format="pdf")

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
plt.title(f"SqueezeNet CNN")
plt.legend()
plt.savefig(f"ValidationSqueezeNetepoch{nb_epochs}v1.pdf", format="pdf")