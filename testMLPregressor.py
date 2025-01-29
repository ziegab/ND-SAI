import torch
from torch import nn
from torch import optim
import csv
import sys
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset

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
# print(numevents)

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
train, val = random_split(dipho_dataset, [400, 26])
train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(val, batch_size=32, shuffle=False)

print(type(tensor_labels_norm))
print(torch.isnan(tensor_inputs).any())  # If True, NaNs exist
print(torch.isinf(tensor_inputs).any())  # If True, Infs exist

# for tensor_inputs, tensor_labels in train_loader:
#     print(tensor_inputs.shape, tensor_labels.shape)
#     break

# Define my model
model = nn.Sequential(
    nn.Flatten(),

    # 1st hidden layer
    nn.Linear((15 * 15) + 2, 256),
    nn.ReLU(),
    # nn.Dropout(p=0.2),

    # 2nd hidden layer
    nn.Linear(256, 256),
    nn.ReLU(),
    # nn.Dropout(p=0.2),

    # 3rd hidden layer
    nn.Linear(256, 128),
    nn.ReLU(),

    # output layer
    nn.Linear(128, 1)
)

# Define my optimizer
params = model.parameters()
optimizer = optim.SGD(params, lr=1e-2)

# Define my loss
loss = nn.MSELoss()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)  # Ensure biases start at zero

model.apply(init_weights)

# My training and validation loops
nb_epochs = 30 # Defines number of times the nn goes through the whole data set. Each time should minimize the loss
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

  print(f'Epoch {epoch +1}, training loss: {torch.tensor(losses).mean():.2f}')

model.eval()
test_images, test_labels_norm = next(iter(val_loader))
test_labels_norm = test_labels_norm.float().unsqueeze(1)
outputs_norm = model(test_images)

outputs = outputs_norm * target_std + target_mean
test_labels = test_labels_norm * target_std + target_mean

print("Target range:", torch.min(test_labels), torch.max(test_labels))
print("Prediction range:", torch.min(outputs), torch.max(outputs))
# print(outputs.shape, test_labels.shape)
for i in range(10):
   print(f"True: {test_labels[i].item():.2f} \tPredicted: {outputs[i].item():.2f}")