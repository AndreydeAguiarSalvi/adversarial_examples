import tqdm
import torch
import torch.nn as nn
from torchattacks import *
from utils.models2 import *
from torch.utils.data import DataLoader


def create_argparser() -> dict:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--model', type=str, default='ALEX', choices=['ALEXNET', 'VGG16', 'VGG19', 'RESNET18', 'RESNET50'], help='which model to train')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100'], help='which dataset to train/eval/attack')
    parser.add_argument('--batchnorm', type=str, default='BatchNorm', choices=['BatchNorm', 'Identity', 'InstanceNorm'], help='mainting or remove batch normalization')
    parser.add_argument('--dropout', type=str, default='Dropout', choices=['Dropout', 'Identity'], help='mainting or remove drop-out')
    parser.add_argument('--attack', type=str, default='FGSM', choices=['FGSM', 'DeepFool', 'BIM', 'CW', 'RFGSM', 'PSG', 'PGD', 'APGD', 'FFGSM', 'TPGD'], help='which adversarial example to use')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum to Stochastic Gradient Descendent')
    parser.add_argument('--epochs_decay', nargs='*', type=int, help='kind of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=.5, help='gamma used in learning rate decay')
    parser.add_argument('--try', type=str, default='0', help='number of attempt')
    parser.add_argument('--device', help='device id (i.e. 0 or 0,1 or cpu)')
    args = vars(parser.parse_args())

    return args


def create_model(args: dict) -> nn.Module:
    if args['model'] == 'ALEXNET': return AlexNet(args).to(args['device'])
    elif 'VGG' in args['model']: return VGG(args).to(args['device'])
    elif 'ResNet' in args['model']: return ResNet(args).to(args['device'])
    else: return None


def create_folder(args: dict):
    import os
    if not os.path.exists(args['model'] + os.sep + args['dataset'] + os.sep + args['attack'] + os.sep + args['try']):
        os.makedirs(args['model'] + os.sep + args['dataset'] + os.sep + args['attack'] + os.sep + args['try'])
    args['folder'] = args['model'] + os.sep + args['dataset'] + os.sep + args['attack'] + os.sep + args['try'] + os.sep


def evaluate(model: nn.Module, criterion: nn.CrossEntropyLoss, args: dict, loader: DataLoader) -> dict:
    result = {'total_acc': .0}
    
    model.eval()

    num_classes = len(args['classes'])
    mloss = torch.zero(1).to(args['device'])
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))  # progress bar
        for i, (images, labels) in pbar:
            images, labels = images.to(args['device']), labels.to(args['device'])
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
    
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            mloss = (mloss * i + loss.item()) / (i + 1)
            s = ('%10s' * 2 + '%10.3g' * 3) % ('%g/%g' % (i, len(loader) - 1), mem, mloss, len(labels) )
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            pbar.set_description(s)

    for i in range(num_classes):
        result[ args['classes'][i] ] = 100 * class_correct[i] / class_total[i]
        result['total_acc'] += class_correct[i] / class_total[i]
    result['loss'] = mloss

    return result


def attack(model: nn.Module, criterion: nn.CrossEntropyLoss, args: dict, loader: DataLoader):
    if has_attack:
        if args['attack'] == 'FGSM':
            attack = FGSM(model, eps=args['epsilon'])