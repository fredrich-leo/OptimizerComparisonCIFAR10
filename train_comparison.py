# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

'''Train CIFAR10 with PyTorch.'''
import json

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

from adabelief import AdaBelief
from lamb import Lamb
from lars import LARS

print("Cuda: ", torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier(m.weight.data)
        torch.nn.init.torch.nn.init.xavier(m.bias.data)


models = {"resnet18": models.resnet18,
          "resnet101": models.resnet101,
          "resnet152": models.resnet152}
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=2048, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=2048, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model


criterion = nn.CrossEntropyLoss()


def test(epoch):
    global best_acc, net, testloader, criterion
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100. * correct / total
    print(f"Test acc {acc}  Loss {test_loss / total}")
    return test_loss / total, acc


optimizers = [(AdaBelief,  5e-3), (Lamb, 5e-3), (optim.Adam, 5e-3), (optim.SGD, 0.1)]

epochs = 50
#lr = 0.005
print(50*len(trainloader))
model_names = ["resnet152", "resnet101", "resnet18"]

data = dict()
for model_name in model_names:
    pbar = tqdm(total=epochs * len(trainloader) * len(optimizers))
    net = models[model_name]()
    state_dict = net.state_dict().copy()
    for optimizer_func, lr in optimizers:
        print("\nTraining " + model_name + " with", optimizer_func.__name__)
        print('==> Building model..')
        net = models[model_name]()
        net.load_state_dict(state_dict)
        net = net.to(device)
        optimizer = optimizer_func(net.parameters(), lr=lr)

        data[model_name + "_" + optimizer_func.__name__] = dict()
        data[model_name + "_" + optimizer_func.__name__]["epochs"] = list()
        data[model_name + "_" + optimizer_func.__name__]["epoch_loss"] = list()
        data[model_name + "_" + optimizer_func.__name__]["test_loss"] = list()
        data[model_name + "_" + optimizer_func.__name__]["test_acc"] = list()
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                pbar.update(1)
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            epoch_loss = train_loss / total
            print(f"\nEpoch {epoch} Train-Loss {epoch_loss}")
            test_loss, test_acc = test(epoch)
            data[model_name + "_" + optimizer_func.__name__]["epochs"].append(epoch)
            data[model_name + "_" + optimizer_func.__name__]["epoch_loss"].append(epoch_loss)
            data[model_name + "_" + optimizer_func.__name__]["test_loss"].append(test_loss)
            data[model_name + "_" + optimizer_func.__name__]["test_acc"].append(test_acc)
    json.dump(data, open(model_name + "_data.json", "w"))
    pbar.close()

for graph in ["epoch_loss", "test_loss", "test_acc"]:
    for model_name in model_names:
        fig = go.Figure()
        for optimizer_func, lr in optimizers:
            fig.add_trace(go.Scatter(x=data[model_name + "_" + optimizer_func.__name__]["epochs"],
                                     y=data[model_name + "_" + optimizer_func.__name__][graph],
                                     mode='lines',
                                     name=optimizer_func.__name__))
        fig.write_html(model_name + "_" + graph + "_fig.html")
