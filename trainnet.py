import numpy as np                   # for matrices
import torch                         # PyTorch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as utils
import torch.nn as nn
import time
import copy

from datetime import datetim
import io
import os
from util import *
import sys
from dataset import *



NETS = {
    'resnet18': resnet18,
    'test_classifier': resnet34
}
OPTIMIZERS = {
    'Adam': optim.Adam,
    'AdamW': optim.AdamW
}

BATCH_SIZE = 100
N_EPOCHS = 15
LEARNING_RATE = 0.0003
NET_NAME = '#'
if NET_NAME != '#':
    MODEL_FNAME = 'model_param_saves/' + args.load
else:
    MODEL_FNAME = '#'
PARAM_DIR = 'model_param_saves/'
LOG_DIR = 'train_logs/'
NORMALIZE = True
ERASE = False


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

business_path = "yelp_dataset/yelp_academic_dataset_business.json"
pure_dataset = CustomDataset(bus_path = business_path)


if torch.cuda.is_available():
    dataset.cuda()


train_sampler = SubsetRandomSampler(np.arange(len(pure_dataset), dtype=np.int64))
#val_sampler = SubsetRandomSampler(np.arange(val_size, dtype=np.int64))



def get_train_loader(batch_size, LA_train):
    train_loader = torch.utils.data.DataLoader(LA_train, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=2, pin_memory=True)
    return train_loader


def createLossAndOptimizer(net, learning_rate=0.001, optimizer='Adam'):
    loss = torch.nn.MSELoss()
    optimizer = OPTIMIZERS[optimizer](net.parameters(), lr=learning_rate)
    
    return loss, optimizer



def trainNet(LA_train, net, batch_size, n_epochs, learning_rate):
    current = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file1 = open(LOG_DIR + "model-history-" + current + ".txt","w")
    loaded_model = MODEL_FNAME if MODEL_FNAME != None else 'None'

    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("loaded_model=", loaded_model)
    print("net_name=", NET_NAME)
    print("normalize=", NORMALIZE)
    print("erase=", ERASE)
    print("=" * 30)

    file1.write("===== HYPERPARAMETERS =====\n")  
    file1.write("batch_size="+ str(batch_size) + "\n")
    file1.write("epochs=" + str(n_epochs) + "\n")
    file1.write("learning_rate=" + str(learning_rate) + "\n")
    file1.write("loaded_model=" + loaded_model + "\n")
    file1.write("net_name=" + NET_NAME + "\n")
    file1.write("normalize=" + str(NORMALIZE) + "\n")
    file1.write("erase=" + str(ERASE) + "\n")
    file1.write("=" * 30 + "\n")
    
    train_loader = get_train_loader(batch_size, pure_dataset)
    n_batches = len(train_loader)
    
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    training_start_time = time.time()
    
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            
            
            optimizer.zero_grad()
            
            outs = net(inputs)
            outputs = torch.squeeze(outs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            running_loss += loss_size.data
            total_train_loss += loss_size.data
            
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                file1.write("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                file1.write("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time) +"\n")
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        total_val_loss = 0
        for inputs, labels in val_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            
            val_outputs = net(inputs)
            if NET_NAME == 'GoogLeNet':
                val_outputs = val_outputs.logits
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        file1.write("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)) + "\n")
        torch.save(net.state_dict(), PARAM_DIR + "model-" + current + '-epoch:' + str(epoch + 1) + ".params")
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    file1.write("Training finished, took {:.2f}s".format(time.time() - training_start_time) + "\n\n")
    file1.close()


from datetime import datetime
print(datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))


CNN = NETS['resnet18']()
if torch.cuda.is_available():
    CNN = CNN.cuda()
if MODEL_FNAME != '#':
    CNN.load_state_dict(torch.load(MODEL_FNAME))
CNN = trainNet(pure_dataset, CNN, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE)
