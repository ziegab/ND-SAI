import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import csv
import sys
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random

# Creating an MLP regressor with the goal of regressing the boost of an AtoGG decay
# Importing information from csv file with each row = label, eta, phi, 15x15 flattened pixel image

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

target_mean, target_std = tensor_labels.mean(), tensor_labels.std()
tensor_labels_norm = (tensor_labels - target_mean) / target_std
            
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

dipho_dataset = CustomDataset(tensor_inputs, tensor_labels_norm)
train, val = random_split(dipho_dataset, [int(numevents*0.7), numevents-int(numevents*0.7)])
train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(val, batch_size=32, shuffle=False)

print(type(tensor_labels_norm))
print(torch.isnan(tensor_inputs).any())  # If True, NaNs exist
print(torch.isinf(tensor_inputs).any())  # If True, Infs exist

# for tensor_inputs, tensor_labels in train_loader:
#     print(tensor_inputs.shape, tensor_labels.shape)
#     break

n_layers = 4
neurons = 512
dropoutpercent = 0.3
# Define my model
model = nn.Sequential(
    nn.Flatten(),

    # 1st hidden layer
    nn.Linear((15 * 15) + 2, neurons),
    nn.BatchNorm1d(neurons),
    nn.ReLU(),
    nn.Dropout(p=dropoutpercent),

    # 2nd hidden layer
    nn.Linear(neurons, neurons),
    nn.BatchNorm1d(neurons),
    nn.ReLU(),
    nn.Dropout(p=dropoutpercent),

    # 3rd hidden layer
    nn.Linear(neurons, neurons),
    nn.BatchNorm1d(neurons),
    nn.ReLU(),
    nn.Dropout(p=dropoutpercent),

    # 4th hidden layer
    nn.Linear(neurons, neurons),
    # nn.BatchNorm1d(neurons),
    nn.ReLU(),
    nn.Dropout(p=dropoutpercent),

    # # 5th hidden layer
    # nn.Linear(neurons, neurons),
    # # nn.BatchNorm1d(128),
    # nn.ReLU(),
    # nn.Dropout(p=dropoutpercent),

    # output layer
    nn.Linear(neurons, 1)
)

# Define my optimizer
params = model.parameters()
optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)
# optimizer = optim.Adam(params, lr=1e-3)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
# optimizer = optim.SGD(params, lr=1e-3)

# Define my loss
loss = nn.MSELoss()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)  # Ensure biases start at zero

model.apply(init_weights)


traininglosses = []
validationlosses = []
# My training and validation loops
nb_epochs = 50 # Defines number of times the nn goes through the whole data set. Each time should minimize the loss
for epoch in range(nb_epochs):
  losses = list()
  model.train()
  for images, labels in train_loader: # getting information from however we load in the information
    # x, y = batch # x is input (a 28*28 image), y is label --> extracted from the batch

    # x: batchsize x 1channel x (28 x 28) --> send this stuff through a linear layer, so need to convert this into a long vector # color image would have 3 channels instead of 1 ???
    # b = x.size(0)
    # x = x.view(b, -1)

    # 1. forward
    l = model(images) # expected output, l:logits

    #2. compute the objective function
    # measuring how well the network does its task of classifying
    J = loss(l, labels.unsqueeze(1).float()) #this part uses a lot of memory since it's plotting stuff

    #3. cleaning the gradients
    model.zero_grad()
    # optimizer.zero_grad()
    # params.grad.zero_()

    #4. accumulate the partial derivatives of J with respect to the parameters
    J.backward() # accumulates the new gradients to the previous ones
    # params.grad.add_(dJ/dparams)

    #5. step in the opposite direction of the gradient --> learning part
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    # or could have done: with torch.no_grad(): params = params - eta * params.grad

    losses.append(J.item())

  print(f'Epoch {epoch +1}, training loss: {torch.tensor(losses).mean():.2f}')
  traininglosses.append(torch.tensor(losses).mean())

  losses = list()
  for images, labels in val_loader: # getting information from however we load in the information
    # x, y = batch # x is input (a 28*28 image), y is label --> extracted from the batch

    # x: batchsize x 1channel x (28 x 28) --> send this stuff through a linear layer, so need to convert this into a long vector # color image would have 3 channels instead of 1 ???
    # b = x.size(0)
    # x = x.view(b, -1)

    # 1. forward
    with torch.no_grad():
      l = model(images) # expected output, l:logits

    #2. compute the objective function
    # measuring how well the network does its task of classifying
    J = loss(l, labels.unsqueeze(1).float()) #this part uses a lot of memory since it's plotting stuff

    losses.append(J.item())

  print(f'Epoch {epoch +1}, validation loss: {torch.tensor(losses).mean():.2f}')
  validationlosses.append(torch.tensor(losses).mean())

# labels2 = []
# inputs2 = []
# with open(sys.argv[2]) as csvfile:
#     reader2 = csv.reader(csvfile)
#     n2 = 0
#     for row in reader2:
#         labels2.append(float(row[0]))
#         tempinput2 = []
#         for i in range(1, len(row)):
#             tempinput2.append(float(row[i]))
#         inputs2.append(tempinput2)

# tensor_labels2 = torch.tensor(labels2, dtype=torch.float32)
# tensor_inputs2 = torch.tensor(inputs2, dtype=torch.float32)
# numevents2 = len(labels2)
# target_mean2, target_std2 = tensor_labels2.mean(), tensor_labels2.std()
# tensor_labels_norm2 = (tensor_labels2 - target_mean2) / target_std2

# dipho_dataset_newmass = CustomDataset(tensor_inputs2, tensor_labels_norm2)
# newmass_loader = DataLoader(dipho_dataset_newmass, batch_size=32, shuffle=False)

model.eval()
# test_images, test_labels_norm = next(iter(newmass_loader))
test_images, test_labels_norm = next(iter(train_loader))
test_labels_norm = test_labels_norm.float().unsqueeze(1)
outputs_norm = model(test_images)

# outputs = outputs_norm * target_std2 + target_mean2
# test_labels = test_labels_norm * target_std2 + target_mean2
outputs = outputs_norm * target_std + target_mean
test_labels = test_labels_norm * target_std + target_mean

print("Target range:", torch.min(test_labels), torch.max(test_labels))
print("Prediction range:", torch.min(outputs), torch.max(outputs))
# print(outputs.shape, test_labels.shape)
x = []
y = []
for i in range(32):
  # print(f"True: {test_labels[i].item():.2f} \tPredicted: {outputs[i].item():.2f}")
  x.append(test_labels[i].item())
  y.append(outputs[i].item())


# # predicted - true /true
plt.scatter(x, y, c='blue', alpha=0.5)
plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
plt.title('True vs. Predicted Boost for MLP Regressor')
plt.savefig(f"LatestPredTruePlot.pdf")

print(f"Printing Neurons={neurons} plots.")
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
plt.title(f"{n_layers} Layers w/ BatchNorm1D, {neurons} Neurons")
plt.legend()
plt.savefig(f"Layers{n_layers}neurons{neurons}allweightdecaybatchnorm3dropout0p3epoch{nb_epochs}val0p3shuffle.pdf", format="pdf")

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
plt.title(f"{n_layers} Layers w/ BatchNorm1D, {neurons} Neurons")
plt.legend()
plt.savefig(f"ValidationLayers{n_layers}neurons{neurons}allweightdecaybatchnorm3dropout0p3epoch{nb_epochs}val0p3shuffle.pdf", format="pdf")


# newvar=[]
# for i in range(32):
#   newvar.append((y[i]-x[i])/x[i])
# plt.hist(newvar, bins=30)
# plt.show()

# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for batch_X, batch_y in train_loader:
#         batch_preds = model(batch_X)  # Get predictions
#         all_preds.append(batch_preds.numpy())
#         all_labels.append(batch_y.numpy())

