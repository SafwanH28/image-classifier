import os
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from PIL import Image
import futility
import fmodel

arg_parse = argparse.ArgumentParser(
    description = 'Training Parser'
)
arg_parse.add_argument('data_dir', action="store", default="./flowers/")
arg_parse.add_argument('--save_dir', action="store", default="./checkpoint.pth")
arg_parse.add_argument('--arch', action="store", default="vgg16")
arg_parse.add_argument('--gpu', action="store", default="gpu")
arg_parse.add_argument('--learning_rate', action="store", default=0.001, type=float, )
arg_parse.add_argument('--hidden_units', action="store", dest="hidden_units", default=512, type=int,)
arg_parse.add_argument('--output_size', action="store", dest="output_size", default=102, type=int,)
arg_parse.add_argument('--epochs', action="store", default=3, type=int)


args = arg_parse.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
gpu = args.gpu
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
output_size = args.hidden_units
epochs = args.epochs

if torch.cuda.is_available() and gpu == 'gpu':
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def main():
    train_loader, valid_loader, test_loader, train_datasets = futility.load_data(data_dir)
    model, criterion = fmodel.setup_network(struct,hidden_units,output_size,lr,gpu)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    # Train Model
    steps = 0
    running_loss = 0
    print_every = 10
    print("Training Model:")
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
          
            if torch.cuda.is_available() and gpu =='gpu':
                inputs, labels = inputs.to(device), labels.to(device)
                model = model.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch ({epoch+1} of {epochs}) "
                      f"Loss: {running_loss/print_every:.3f} "
                      f"Validation Loss: {valid_loss/len(valid_loader):.3f} "
                      f"Accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
    
    model.class_to_idx =  train_datasets.class_to_idx
    torch.save({'structure' :struct,
                'hidden_units':hidden_units,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx}, 
                save_dir)
    print("Checkpoint Has Been Saved")
if __name__ == "__main__":
    main()               
