import torch
import random
import numpy as np
import torchvision.datasets as D
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_Dataset(args, is_train=True, is_test=True, train_transform=None, test_transform=None):
    if args['dataset'] == 'CIFAR10': return get_CIFAR10(args, is_train, is_test, train_transform, test_transform)
    elif args['dataset'] == 'CIFAR100': return get_CIFAR100(args, is_train, is_test, train_transform, test_transform)
    elif args['dataset'] == 'MNIST': return get_MNIST(args, is_train, is_test, train_transform, test_transform)


def get_CIFAR10(args, is_train=True, is_test=True, train_transform=None, test_transform=None):
    train_loader, valid_loader, test_loader, classes = None, None, None, None
    # Data Augmentation
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1*random.uniform(0, 1),
                contrast=0.1*random.uniform(0, 1),
                saturation=0.1*random.uniform(0, 1),
                hue=0.1*random.uniform(0, 1)
            ),
            transforms.Resize(64, 64),
            transforms.ToTensor(),
            transforms.Normalize(
                (.491399679, .482158408, .446530914), 
                (.247032232, .243485128, .261587842)
            )
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(64, 64),
            transforms.ToTensor(),
            transforms.Normalize(
                (.491399679, .482158408, .446530914), 
                (.247032232, .243485128, .261587842)
            )
        ])
    
    # Data sets
    if is_train:
        train_set = D.CIFAR10(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        valid_set = D.CIFAR10(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        # Spliting train in train/validation
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(.3 * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
    if is_test:
        test_set = D.CIFAR10(
            root='./data', train=False,
            download=True, transform=test_transform
        )
    
        
    if is_train:
        train_loader = DataLoader(
            train_set, batch_size=args['batch_size''],
            num_workers=4,
            sampler=train_sampler
        )
        valid_loader = DataLoader(
            valid_set, batch_size=args['batch_size''],
            num_workers=4,
            sampler=valid_sampler
        )
    if is_test: 
        test_loader = DataLoader(
            test_set, batch_size=args['batch_size''],
            num_workers=4
        )
    # Classes names
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    args['classes'] = classes
    return train_loader, valid_loader, test_loader


def get_CIFAR100(args, is_train=True, is_test=True, train_transform=None, test_transform=None):
    train_loader, valid_loader, test_loader, classes = None, None, None, None
    # Data Augmentation
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1*random.uniform(0, 1),
                contrast=0.1*random.uniform(0, 1),
                saturation=0.1*random.uniform(0, 1),
                hue=0.1*random.uniform(0, 1)
            ),
            transforms.Resize(64, 64),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            )
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(64, 64),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            )
        ])
    
    # Data sets
    if is_train:
        train_set = D.CIFAR10(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        valid_set = D.CIFAR10(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        # Spliting train in train/validation
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(.3 * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
    if is_test:
        test_set = D.CIFAR10(
            root='./data', train=False,
            download=True, transform=test_transform
        )
    
        
    if is_train:
        train_loader = DataLoader(
            train_set, batch_size=args['batch_size''],
            num_workers=4,
            sampler=train_sampler
        )
        valid_loader = DataLoader(
            valid_set, batch_size=args['batch_size''],
            num_workers=4,
            sampler=valid_sampler
        )
    if is_test: 
        test_loader = DataLoader(
            test_set, batch_size=args['batch_size''],
            num_workers=4
        )
    # Classes names
    args['classes'] = [i for i in range(100)]
    return train_loader, valid_loader, test_loader, classes   


def get_MNIST(args, is_train=True, is_test=True, train_transform=None, test_transform=None):
    train_loader, valid_loader, test_loader, classes = None, None, None, None
    
    # Data Augmentation        
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.Resize(64, 64),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(64, 64),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if is_train:
        # Data sets
        train_set = D.MNIST(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        valid_set = D.MNIST(
            root='./data', train=True, 
            download=True, transform=train_transform
        )
        # Spliting train in train/validation
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(.3 * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
    if is_test:
        test_set = D.MNIST(
            root='./data', train=False,
            download=True, transform=test_transform
        )
    
    if is_train:
        train_loader = DataLoader(
            train_set, batch_size=args['batch_size''],
            num_workers=4,
            sampler=train_sampler
        )
        valid_loader = DataLoader(
            valid_set, batch_size=args['batch_size''],
            num_workers=4,
            sampler=valid_sampler
        )
    if is_test:
        test_loader = DataLoader(
            test_set, batch_size=args['batch_size''],
            num_workers=4
        )
    # Classes names
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9') 
    args['classes'] = classes
    return train_loader, valid_loader, test_loader