'''
Created Date: Wednesday, July 27th 2022, 1:06:27 pm
Author: Rutuja Gurav (rutuja.gurav@email.ucr.edu)
Copyright (c) 2022 M.A.D. Lab @ UCR (https://madlab.cs.ucr.edu)

'''

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=32):
    # Use transforms.compose method to reformat images for modeling,
    # and save to variable all_transforms for later use
    all_transforms = transforms.Compose([
                                        # transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                            std=[0.2023, 0.1994, 0.2010])
                                        ])
    # Create Training dataset
    dataset_ = torchvision.datasets.CIFAR10(root = './data',
                                train = True,
                                transform = all_transforms,
                                download = True)
    val_size = 5000
    train_size = len(dataset_) - val_size
    train_dataset, val_dataset = random_split(dataset_, [train_size, val_size])

    # Create Testing dataset
    test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                train = False,
                                transform = all_transforms,
                                download=True)

    # Instantiate loader objects to facilitate processing
    train_loader = DataLoader(dataset = train_dataset,
                                batch_size = batch_size,
                                shuffle = True)
    
    val_loader = DataLoader(dataset = val_dataset,
                                batch_size = batch_size,
                                shuffle = False)

    test_loader = DataLoader(dataset = test_dataset,
                                batch_size = batch_size,
                                shuffle = False)
    
    return train_loader, val_loader, test_loader, (len(train_dataset), len(val_dataset), len(test_dataset))