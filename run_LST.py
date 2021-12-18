import argparse
import math
import time

import torch
import torch.nn as nn
import LSTNet
import numpy as np
import importlib

import pickle
import data_loader


def train(model, device, train_loader, optimizer, criterion, epoch,
          train_losses, train_accuracies):
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
    print("Train Epoch: %d Train Loss: %.4f Train Accuracy: %.2f" %
          (epoch, train_loss, accuracy))

# testing model in pytorch


def test(model, device, test_loader, criterion, epoch, test_losses, test_accuracies):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        # test loss calculation
        test_loss = (test_loss/len(test_loader.dataset))
        # calculating the accuracy in the validation step
        accuracy = (100*correct)/len(test_loader.dataset)
        test_accuracies.append(accuracy)
        test_losses.append(test_loss)
        # logging the results
        print("Test Epoch: %d Test Loss: %.4f Test Accuracy: %.2f" %
              (epoch, test_loss, accuracy))

# model fitting in pytorch


def fit(model, device, train_loader, test_loader, optimizer, criterion, no_of_epochs):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(0, no_of_epochs):
        train(model, device, train_loader, optimizer,
              criterion, epoch, train_losses, train_accuracies)
        test(model, device, test_loader, criterion,
             epoch, test_losses, test_accuracies)
        return train_losses, test_losses, train_accuracies, test_accuraciesrgparse.ArgumentParser(description='PyTorch Time series forecasting')


parser = argparse.ArgumentParser(description='PyTorch Time Series Forecasting')
parser.add_argument('--train', type=str, required=True,
                    help='location of the training data file')
parser.add_argument('--test', type=str, required=True,
                    help='location of the test data file')
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=100,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=24 * 7,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

f = open("data.pkl")
dataset = pickle.load(f)

print(dataset[2014])

trainset = None
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = None
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('Buy', 'Hold', 'Sell')

net = LSTNet(args, num_parameters)
net = net.to(device)
if device == 'cuda':
    net = nn.DataParallel(net)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200)

args.cuda = args.gpu is not None
if args.cuda:
    torch.device('cuda')
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 0.6, 0.2, args.cuda,
                    args.horizon, args.window, args.normalize)
print(Data.rse)

model = eval(args.model).Model(args, Data)

if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False)
else:
    criterion = nn.MSELoss(size_average=False)
evaluateL2 = nn.MSELoss(size_average=False)
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()


best_val = 10000000
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training')
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(
            Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        val_loss, val_rae, val_corr = evaluate(
            Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.

        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 5 == 0:
            test_acc, test_rae, test_corr = evaluate(
                Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(
                test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
test_acc, test_rae, test_corr = evaluate(
    Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(
    test_acc, test_rae, test_corr))
