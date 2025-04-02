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

# Previous Boost CNN with the goal of regressing the boost of an AtoGG decay
# Importing information from csv file with each row = label, eta, phi, 15x15 flattened pixel image

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        # self.labels = (labels - labels.mean()) / labels.std()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

def split_train_val(arg):
    labels = []
    inputs = []
    etas = []
    with open(arg) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(f"Row: {len(row)}")
            labels.append(float(row[0]))
            etas.append((float(row[2])*6)-3)
            tempinput = []
            for i in range(4, len(row)):
                tempinput.append(float(row[i]))
            # print(f"Extracted input: {len(tempinput)}")
            inputs.append(tempinput)
    numevents = len(labels)
    # Pair elements together and shuffle
    combined = list(zip(labels, inputs))
    random.shuffle(combined)
    split_idx = int(len(combined) * 0.7)
    # print(split_idx)
    train_combined = combined[:split_idx]
    val_combined = combined[split_idx:]
    # print(f"Length of train_combined: {len(train_combined)}")
    # print(f"Length of val_combined: {len(val_combined)}")
    # Unzip back into separate lists
    train_labels, train_inputs = zip(*train_combined)
    val_labels, val_inputs = zip(*val_combined)
    # Convert back to lists (since zip() returns tuples)
    train_labels = list(train_labels)
    train_inputs = list(train_inputs)
    val_labels = list(val_labels)
    val_inputs = list(val_inputs)
    # print(len(train_inputs), len(train_labels))
    tensor_train_inputs = torch.stack([torch.tensor(t, dtype=torch.float32).view(1,input_size,input_size) for t in train_inputs])
    tensor_val_inputs = torch.stack([torch.tensor(t, dtype=torch.float32).view(1,input_size,input_size) for t in val_inputs])
    return train_labels, val_labels, tensor_train_inputs, tensor_val_inputs

labels_train_tracker = []
labels_val_tracker = []
inputs_train_tracker = []
inputs_val_tracker = []
etas_tracker = []
argument_tracker = 0

file_dir = str(sys.argv[1])
print(file_dir)
csv_files = glob.glob(f"{file_dir}/*.csv")

input_size = 32

for arg in (csv_files):
    argument_tracker += 1
    arg_train_labels, arg_val_labels, arg_train_inputs, arg_val_inputs = split_train_val(arg)
    labels_train_tracker.append(arg_train_labels)
    labels_val_tracker.append(arg_val_labels)
    inputs_train_tracker.append(arg_train_inputs)
    inputs_val_tracker.append(arg_val_inputs)
    # if argument_tracker == 3:
    #     break

tensor_train_inputs = [tensor for sublist in inputs_train_tracker for tensor in sublist]
tensor_val_inputs = [tensor for sublist in inputs_val_tracker for tensor in sublist]
tensor_train_labels = [torch.tensor([inner]) for outer in labels_train_tracker for inner in outer]
tensor_val_labels = [torch.tensor([inner]) for outer in labels_val_tracker for inner in outer]

# print(len(tensor_train_inputs), len(tensor_train_labels))

flattened = torch.cat(tensor_train_labels)
print(f"Length of flattened train_labels: {len(flattened)}")
target_mean, target_std = flattened.mean(), flattened.std()
norm_labels_train_tracker = [(t-target_mean)/target_std for t in tensor_train_labels]
norm_labels_val_tracker = [(t-target_mean)/target_std for t in tensor_val_labels]
print(target_mean, target_std)

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

combined_tensors = list(zip(norm_labels_train_tracker, tensor_train_inputs))
random.shuffle(combined_tensors)
norm_labels_train_tracker, tensor_train_inputs = map(list, zip(*combined_tensors))
combined_val_tensors = list(zip(norm_labels_val_tracker, tensor_val_inputs))
random.shuffle(combined_val_tensors)
norm_labels_val_tracker, tensor_val_inputs = map(list, zip(*combined_val_tensors))

train_dataset = CustomDataset(torch.cat(tensor_train_inputs), torch.cat(norm_labels_train_tracker))#, transform=transform)
val_dataset = CustomDataset(torch.cat(tensor_val_inputs), torch.cat(norm_labels_val_tracker))#, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Adjusted Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # Smaller kernel for smaller input
        self.bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.0)  
        
        self.fc1 = nn.Linear(32 * 8 * 8, 784)  # Adjusted input size
        self.bn1 = nn.BatchNorm1d(784)
        self.fc2 = nn.Linear(784, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu((self.conv1(x)))  # Conv1
        x = F.relu((self.conv2(x)))  # Conv2
        x = self.pool(x)           # MaxPool1
        x = F.relu((self.conv3(x)))  # Conv3
        x = F.relu((self.conv4(x)))  # Conv4
        x = self.pool2(x)
        
        # x = x.view(x.size(0), -1)  # Flatten
        # x = self.dropout( F.relu((self.fc1(x))))    # FC1
        # x = self.dropout(F.relu((self.fc2(x))))    # FC2
        # x = F.relu((self.fc3(x)))    # FC3
        # x = self.fc4(x)            # Output
        
        # x = F.relu(self.bn(self.conv1(x)))  # Conv1
        # x = F.relu(self.bn(self.conv2(x)))  # Conv2
        # x = self.pool(x)           # MaxPool1
        # x = F.relu(self.bn(self.conv3(x)))  # Conv3
        # x = F.relu(self.bn(self.conv4(x)))  # Conv4
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout( F.relu(self.bn1(self.fc1(x))))    # FC1
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))    # FC2
        x = F.relu(self.bn3(self.fc3(x)))    # FC3
        x = self.fc4(x)            # Output
        
        return x
    
# Training model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = CNNModel()

# Define Loss and Optimizer
criterion = nn.MSELoss()  # Regression loss function
# criterion = nn.HuberLoss(delta=1.0)
# criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)  # Ensure biases start at zero

model.apply(init_weights)

# Training Loop
nb_epochs = 40
traininglosses = []
validationlosses = []
for epoch in range(nb_epochs):
    model.train()
    # running_loss = 0.0
    losses = list()
    for images, labels in train_loader:  # Use your dataset loader
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
    for images, labels in val_loader: 
        # 1. forward
        with torch.no_grad():
            l = model(images.unsqueeze(1)) 

        #2. compute the objective function
        loss = criterion(l, labels.view(-1, 1)) 

        losses.append(loss.item())

    print(f'Epoch {epoch +1}, validation loss: {torch.tensor(losses).mean():.2f}')
    validationlosses.append(torch.tensor(losses).mean())

model.eval()
x_train = []
y_train = []
with torch.no_grad():
    # for i,loader in enumerate(trains):
        for test_images, test_labels_norm in train_loader:
            test_labels_norm = test_labels_norm.unsqueeze(1)
            outputs_norm = model(test_images.unsqueeze(1))
            outputs = (outputs_norm * target_std) + target_mean
            test_labels = (test_labels_norm * target_std) + target_mean
            # print("Target range:", torch.min(test_labels), torch.max(test_labels))
            # print("Prediction range:", torch.min(outputs), torch.max(outputs))
            for j in range(32):
                if 0 < outputs[j].item() < 1500:
                    x_train.append(test_labels[j].item())
                    y_train.append(outputs[j].item())
        # if i == 0:
        #     break

x_val = []
y_val = []
with torch.no_grad():
    # for i,loader in enumerate(vals):
        for test_images, test_labels_norm in val_loader:
            test_labels_norm = test_labels_norm.unsqueeze(1)
            outputs_norm = model(test_images.unsqueeze(1))
            outputs = (outputs_norm * target_std) + target_mean
            test_labels = (test_labels_norm * target_std) + target_mean
            # print("Target range:", torch.min(test_labels), torch.max(test_labels))
            # print("Prediction range:", torch.min(outputs), torch.max(outputs))
            for j in range(32):
                if 0 < outputs[j].item() < 1500:
                    x_val.append(test_labels[j].item())
                    y_val.append(outputs[j].item())
        # if i == 0:
        #     break

# # predicted - true /true
plt.figure(1)
plt.scatter(x_train, y_train, c='blue', alpha=0.6, s=1)
plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
plt.title('True vs. Predicted Boost for Previous Boost CNN (Training Data)')
plt.savefig(f"TrainPredTrueBoost.pdf")

plt.figure(2)
plt.scatter(x_val, y_val, c='blue', alpha=0.6, s=1)
plt.plot([0, 1000], [0, 1000], color='black', linestyle='-', linewidth=1, label='Predicted = True')
plt.xlabel('True Boost')
plt.ylabel('Predicted Boost')
plt.title('True vs. Predicted Boost for Previous Boost CNN (Validation Data)')
plt.savefig(f"ValPredTrueBoost.pdf")

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
plt.title(f"Previous Boost CNN")
plt.legend()
plt.savefig(f"Boostepoch{nb_epochs}v1.pdf", format="pdf")

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
plt.title(f"Previous Boost CNN")
plt.legend()
plt.savefig(f"ValidationBoostepoch{nb_epochs}v1.pdf", format="pdf")