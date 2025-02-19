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
from itertools import chain

# Creating an MLP regressor with the goal of regressing the boost of an AtoGG decay
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
            for i in range(1, len(row)):
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

    target_mean, target_std = tensor_labels.mean(), tensor_labels.std()
    tensor_labels_norm = (tensor_labels - target_mean) / target_std

    return tensor_labels_norm, tensor_inputs, numevents, target_std, target_mean

datasets = []
numevents_tracker = []
std_tracker = []
mean_tracker = []

for arg in sys.argv[1:]:
    arg_labels, arg_inputs, arg_numevents, arg_std, arg_mean = get_tensor_inputs_labels(arg)
    datasets.append(CustomDataset(arg_inputs, arg_labels))
    numevents_tracker.append(arg_numevents)
    std_tracker.append(arg_std)
    mean_tracker.append(arg_mean)

trains = []
vals = []

for i, dataset in enumerate(datasets):
    numevents = numevents_tracker[i]
    train, val = random_split(dataset, [int(numevents*0.7), numevents-int(numevents*0.7)])
    train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)
    trains.append(train_loader)
    vals.append(val_loader)

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

# Define my loss
loss = nn.MSELoss()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)  # Ensure biases start at zero

model.apply(init_weights)

traininglosses = []
validationlosses = []
nb_epochs = 50 
for epoch in range(nb_epochs):
  losses = list()
  model.train()
  for images, labels in chain(*trains): 
    # 1. forward
    l = model(images) 

    #2. compute the objective function
    J = loss(l, labels.unsqueeze(1).float()) 

    #3. cleaning the gradients
    model.zero_grad()

    #4. accumulate the partial derivatives of J with respect to the parameters
    J.backward() 

    #5. step in the opposite direction of the gradient --> learning part
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    losses.append(J.item())

  print(f'Epoch {epoch +1}, training loss: {torch.tensor(losses).mean():.2f}')
  traininglosses.append(torch.tensor(losses).mean())

  losses = list()
  for images, labels in chain(*vals): 
    # 1. forward
    with torch.no_grad():
      l = model(images) 

    #2. compute the objective function
    J = loss(l, labels.unsqueeze(1).float()) 

    losses.append(J.item())

  print(f'Epoch {epoch +1}, validation loss: {torch.tensor(losses).mean():.2f}')
  validationlosses.append(torch.tensor(losses).mean())


# model.eval()
# x = []
# y = []
# with torch.no_grad():
#     for i,train_loader in enumerate(vals):
#         for test_images, test_labels_norm in train_loader:
#             test_labels_norm = test_labels_norm.float().unsqueeze(1)
#             outputs_norm = model(test_images)
#             outputs = outputs_norm * std_tracker[i] + mean_tracker[i]
#             test_labels = test_labels_norm * std_tracker[i] + mean_tracker[i]
#             # print("Target range:", torch.min(test_labels), torch.max(test_labels))
#             # print("Prediction range:", torch.min(outputs), torch.max(outputs))
#             for j in range(32):
#                 # if outputs[j].item() > 0:
#                     x.append(test_labels[j].item())
#                     y.append(outputs[j].item())


# # # predicted - true /true
# plt.scatter(x, y, c='blue', alpha=0.5)
# plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
# plt.xlabel('True Boost')
# plt.ylabel('Predicted Boost')
# plt.title('True vs. Predicted Boost for MLP Regressor')
# plt.savefig(f"LatestPredTruePlot.pdf")

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
plt.savefig(f"Layers{n_layers}neurons{neurons}epoch{nb_epochs}multi.pdf", format="pdf")

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
plt.savefig(f"ValidationLayers{n_layers}neurons{neurons}epoch{nb_epochs}multi.pdf", format="pdf")

