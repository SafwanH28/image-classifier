import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
from torch.autograd import Variable

arch = {"vgg16":25088, "densenet121":1024}

def load_data(root = "./flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),                                                                                               
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle = True)

    return train_loader, valid_loader, test_loader, train_datasets