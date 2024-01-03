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

def setup_network(structure='vgg16', hidden_units=4096, output_size=102, lr=0.001, device='gpu'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict    

    model.classifier = nn.Sequential(OrderedDict([
                          ('1', nn.Linear(25088, 2048)),
                          ('ReLU', nn.ReLU()),
                          ('2', nn.Linear(2048, 256)),
                          ('ReLU', nn.ReLU()),
                          ('3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    print(model)
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    return model, criterion

def save_checkpoint(train_data, model = 0, save_dir = 'checkpoint.pth', structure = 'vgg16', hidden_units = 4096, output_size = 4096, lr = 0.001, epochs = 1):
    model.class_to_idx =  train_datasets.class_to_idx
    torch.save({'structure' :structure,
                'hidden_units':hidden_units,
                'output_size':output_size,
                'learning_rate':lr,
                'epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                save_dir)
    
def load_checkpoint(save_dir = 'checkpoint.pth'):
    checkpoint = torch.load(save_dir)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    structure = checkpoint['structure']

    model, _ = setup_network(structure, hidden_units, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, topk=5, device='gpu'):   
    model.to('cuda')
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model.forward(img.cuda())
        
    probs = torch.exp(output).data
    return probs.topk(topk)


def process_image(image):
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    image = img_transforms(img_pil)
    return image