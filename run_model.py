import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import importlib

import pickle
import data_loader_helper as dlh

from functools import reduce


def train(model, device, train_loader, optimizer, criterion, epoch,
          train_losses, train_accuracies, log_file):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    # calculating the total loss
    train_loss = ((train_loss)/len(train_loader.dataset))
    train_losses.append(train_loss)
    # calculating the accuracy
    accuracy = (100*correct) / len(train_loader.dataset)
    train_accuracies.append(accuracy)
    # logging the result
    log_file.write("Train Epoch: %d Train Loss: %.4f Train Accuracy: %.2f\n" %
                   (epoch, train_loss, accuracy))

# testing model in pytorch


def test(model, device, test_loader, criterion, epoch, test_losses, test_accuracies, log_file):
    model.eval()
    test_loss = 0
    correct = 0
    sells = []
    with torch.no_grad():
        # Change to data, target if no extra data is passed to TestLoaderHelper
        for data, target, date in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            sells += [date[i].item() for i in range(len(pred)) if pred[i] == 2]
        # test loss calculation
        test_loss = (test_loss/len(test_loader.dataset))
        # calculating the accuracy in the validation step
        accuracy = (100*correct)/len(test_loader.dataset)
        test_accuracies.append(accuracy)
        test_losses.append(test_loss)
        # logging the results
        log_file.write("Test Epoch: %d Test Loss: %.4f Test Accuracy: %.2f\n" %
                       (epoch, test_loss, accuracy))
        return sells

# model fitting in pytorch


def fit(model, device, train_loader, test_loader, optimizer, criterion, no_of_epochs, log_filename, model_filename, csv_filename):
    log_file = open(log_filename, "w")
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(0, no_of_epochs):
        train(model, device, train_loader, optimizer,
              criterion, epoch, train_losses, train_accuracies, log_file)
        sells = test(model, device, test_loader, criterion,
             epoch, test_losses, test_accuracies, log_file)

    with open(model_filename, "wb") as f:
        torch.save(model, f)

    with open(csv_filename, "w") as f:
        f.write(",".join(list(map(str, sells))))

    return train_losses, test_losses, train_accuracies, test_accuracies


def train_and_test(model, model_name, data_pkl="data_lagged.pkl", label_pkl="label_for_matrix.pkl", seed=0xE5A0):
    NUM_EPOCHS = 150

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(data_pkl, "rb") as f:
        dataset = pickle.load(f)

    with open(label_pkl, "rb") as f:
        labels = pickle.load(f)

    for test_year in dataset.keys():
        trainset = [d for y, d in dataset.items() if y != test_year]
        trainset = reduce(lambda x, y: x.append(y), trainset)

        trainlabels = [labels[y] for y, d in dataset.items() if y != test_year]
        trainlabels = reduce(lambda x, y: x + y, trainlabels)

        trainset = dlh.TrainLoaderHelper(np.nan_to_num(trainset.drop(['Date'], axis=1).to_numpy(
        )), np.array(trainlabels), torch.Tensor)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = dataset[test_year]

        testlabels = labels[test_year]

        testset = dlh.TestLoaderHelper(np.nan_to_num(testset.drop(['Date'], axis=1).to_numpy(
        )), np.array(testlabels), torch.Tensor, testset['Days Since Harvest'].to_numpy())
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        classes = ('Buy', 'Hold', 'Sell')

        log_filename = model_name + "-" + str(test_year) + ".log"
        model_filename = model_name + "-" + str(test_year) + ".pt"
        csv_filename = model_name + "-" + str(test_year) + ".csv"

        net = copy.deepcopy(model)
        net = net.to(device)
        if device == 'cuda':
            net = nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200)

        train_losses, test_losses, train_accuracies, test_accuracies = fit(net, device, trainloader, testloader, optimizer,
                                                                           criterion, NUM_EPOCHS, log_filename, 
                                                                           model_filename, csv_filename)
        print("Final train accuracy: %lf, Final test accuracy: %lf" % (train_accuracies[-1], test_accuracies[-1]))
