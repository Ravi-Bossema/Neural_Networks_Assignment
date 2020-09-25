import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from NeuralNetworks.imagenette import get_dataloaders
import timeit


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.linear_1 = nn.Linear(IM_SIZE, H_SIZE)
        self.linear_2 = nn.Linear(H_SIZE, LABELS)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        h = F.relu(self.linear_1(x))
        y = F.softmax(self.linear_2(h))
        return y


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.linear_1 = nn.Linear(IN_SIZE_CNN, H_SIZE)
        self.linear_2 = nn.Linear(H_SIZE, LABELS)
        self.conv = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=FILTERS, kernel_size=KERNEL_SIZE,
                              stride=STRIDE, padding=PADDING)
        self.pool = nn.MaxPool2d(POOLING)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = torch.flatten(x, start_dim=1)
        h = F.relu(self.linear_1(x))
        y = F.softmax(self.linear_2(h))
        return y


def plot_training(x, y_train, y_val):
    """Plots the mean training and validation loss of each epoch"""
    plt.plot(x, y_train, label="Training loss")
    plt.plot(x, y_val, label="Validation loss")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.ylabel("Mean loss")
    plt.xlabel("Epoch number")
    plt.title("Mean loss per epoch using a(n) " + network_type + " with a learning rate of " + str(learning_rate))
    plt.show()


def model(type):
    """Returns a model based on the chosen network_type"""
    if type == "FCN":
        return FCN()
    elif type == "CNN":
        return CNN()
    else:
        print("Please choose a network_type of either FCN or CNN")
        exit()


def train():
    """Trains the model for one epoch and returns the mean training loss"""
    epoch_t_loss = 0
    for batch in range(len(train_loader)):
        x, y = next(iter(train_loader))
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_pred = net(x)

        # Compute loss
        t_loss = objective(y_pred, y)
        epoch_t_loss += t_loss.item()
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
    return epoch_t_loss/TRAIN_LEN


def validate():
    """Validates the model over the entire validation set and returns the
    mean validation loss"""
    epoch_v_loss = 0
    for batch in range(len(valid_loader)):
        val_x, val_y = next(iter(valid_loader))
        val_x = val_x.to(DEVICE)
        val_y = val_y.to(DEVICE)
        val_pred = net(val_x)

        epoch_v_loss += objective(val_pred, val_y).item()
    return epoch_v_loss/VALID_LEN


# -------------------- Setting up --------------------
# Constants
DEVICE = torch.device("cuda")
IM_SIZE = 3072  # 3*32*32
H_SIZE = 300
LABELS = 10

IN_SIZE_CNN = 8192  # 32 * 16 * 16
IN_CHANNELS = 3
FILTERS = 32
STRIDE = 1
KERNEL_SIZE = (3, 2)
PADDING = 1
POOLING = (2, 2)

TRAIN_LEN = 8049  # 85% of the entire training set (size of the actual training set)
VALID_LEN = 1420  # 15% of the entire training set (size of the actual validation set)

# Hyperparameters
learning_rate = 5e-4
batch_size = 32
epochs = 50
network_type = "CNN"  # FCN for fully connected network or CNN for convolutional neural network

# Get Data
train_loader, valid_loader, test_loader = get_dataloaders(batch_size=batch_size)

net = model(network_type)
net.to(DEVICE)

# Define objective function
objective = nn.CrossEntropyLoss(reduction='sum')
# Define Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# -------------------- Training & Validating --------------------
T_loss_list = []
V_loss_list = []

start_time = timeit.default_timer()

for epoch in range(epochs):
    print("Epoch ", epoch+1)
    # Training
    net.train(True)
    t_loss = train()
    T_loss_list.append(t_loss)

    # Validation
    net.train(False)
    v_loss = validate()
    V_loss_list.append(v_loss)

end_time = timeit.default_timer()
duration = end_time - start_time
print("Training duration in seconds: ", duration)
print("Per epoch: ", duration/epochs)

plot_training(torch.arange(start=1, end=epochs+1), T_loss_list, V_loss_list)

# -------------------- Testing --------------------
test_loss = 0
correct = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
net.train(False)
for data, target in test_loader:
    data = data.to(DEVICE)
    target = target.to(DEVICE)
    net_out = net(data)
    # sum up batch loss
    test_loss += objective(net_out, target).item()
    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data).sum()
    c = (pred == target).squeeze()
    for i in range(4):
        label = target[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

# Print the average loss and accuracy
test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# Print the accuracy per class
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        test_loader.dataset.classes[i], 100 * class_correct[i] / class_total[i]))
