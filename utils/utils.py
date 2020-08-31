import tqdm
import torch
import numpy as np
import torch.nn as nn
from torchattacks import *
from utils.models import AlexNet, VGG, ResNet
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_argparser() -> dict:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--model', type=str, default='RESNET18', choices=['MOBILESMALL', 'MOBILELARGE', 'VGG16', 'VGG19', 'RESNET18', 'RESNET50'], help='which model to train')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100'], help='which dataset to train/eval/attack')
    parser.add_argument('--batchnorm', type=str, default='BatchNorm', choices=['BatchNorm', 'Identity', 'InstanceNorm'], help='mainting or remove batch normalization')
    parser.add_argument('--dropout', type=str, default='Dropout', choices=['Dropout', 'Identity'], help='mainting or remove drop-out')
    parser.add_argument('--attack', type=str, default='FGSM', choices=['FGSM', 'DeepFool', 'BIM', 'CW', 'RFGSM', 'PSG', 'PGD', 'APGD', 'FFGSM', 'TPGD'], help='which adversarial example to use')
    parser.add_argument('--eps', type=float, default=None, help='value to use as epsilon. If None, will be a range from 0 to 0.3')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum to Stochastic Gradient Descendent')
    parser.add_argument('--epochs_decay', nargs='*', type=int, help='kind of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=.5, help='gamma used in learning rate decay')
    parser.add_argument('--try', type=str, default='0', help='number of attempt')
    parser.add_argument('--weights', type=str, help='weights to load')
    parser.add_argument('--only_norm', type=lambda x: (str(x).lower() == 'true'), help='if true, freezes other weights and trains only the normalization')
    parser.add_argument('--device', help='device id (i.e. 0 or 0,1 or cpu)')
    args = vars(parser.parse_args())

    return args


def create_model(args: dict) -> nn.Module:
    if 'VGG' in args['model']: return VGG(args).to(args['device'])
    elif 'RESNET' in args['model']: return ResNet(args).to(args['device'])
    elif 'MOBILE' in args['model']: 
        args['model_mode'] = 'LARGE' if 'LARGE' in args['model'] else 'SMALL'
        return MobileNetV3(args).to(args['device'])
    else return None


def create_folder(args: dict):
    import os
    model = args['model']
    if 'RESNET' in model or 'VGG' in model: model += '_norm-' + args['batchnorm']
    if 'ALEX' in model or 'VGG' in model: model += '_drop-' + args['dropout']
    path = model + os.sep + args['dataset'] + os.sep + args['try'] + os.sep
    if not os.path.exists(path):
        os.makedirs(path)
    args['folder'] =  path


def get_folder(args: dict):
    import os
    folder = args['weights'].split(os.sep)
    args['folder'] = ''
    for i in range(len(folder) -1): args['folder'] += folder[i] + os.sep


def evaluate(model: nn.Module, criterion: nn.CrossEntropyLoss, args: dict, loader: DataLoader) -> dict:
    result = {'total_acc': .0}
    
    model.eval()

    num_classes = len(args['classes'])
    mloss = torch.zeros(1).to(args['device'])
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    targets = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))  # progress bar
        for i, (X, Y) in pbar:
            X, Y = X.to(args['device']), Y.to(args['device'])
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            _, predicted = torch.max(Y_hat, 1)
            c = (predicted == Y).squeeze()
    
            for j in range(len(Y)):
                label = Y[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1

            mloss = (mloss * i + loss.item()) / (i + 1)
            targets += len(Y)
            mem = torch.cuda.memory_cached()
            s = ('%10s' + '%10.3g' * 3) % ('%g/%g' % (i, len(loader) - 1), mem, mloss, targets )
            pbar.set_description(s)

    for i in range(num_classes):
        result[ args['classes'][i] ] = 100 * class_correct[i] / class_total[i]
        result['total_acc'] += (100 * class_correct[i] / class_total[i]) / num_classes
    result['loss'] = mloss

    return result


def attack(model: nn.Module, args: dict, loader: DataLoader) -> dict:
    attack = None
    num_classes = len(args['classes'])
    classes = args['classes']
    result = {'total_acc': .0}

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    adv_ex = []
    targets = 0

    model.eval()

    # From https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb
    if args['attack'] == 'FGSM': attack = FGSM(model, eps=args['eps'])

    # From https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Black%20Box%20Attack%20with%20CIFAR10.ipynb
    elif args['attack'] == 'PGD': attack = PGD(model, eps=args['eps'], alpha=2/255, steps=7)

    # All attacks https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20with%20Imagenet.ipynb
    
    print(('\n' + '%10s' * 4) % ('Acc', 'eps', 'Corrects', 'Targets'))
    pbar = tqdm.tqdm(enumerate(loader), total=len(loader))  # progress bar
    for i, (X, Y) in pbar:
        X, Y = X.to(args['device']), Y.to(args['device'])
        
        if args['attack'] in ['FGSM', 'PGD']: 
            if args['eps'] > .0: X = attack(X, Y).to(args['device'])
        
        Y_hat = model(X)
        _, predicted = torch.max(Y_hat.data, 1)
        c = (predicted == Y).squeeze()
        
        if len(adv_ex) < 5:
            denormalize(X, args)
            adv_ex.append((
                classes[Y[0].item()], 
                classes[Y_hat.max(1, keepdim=True)[1][0].item()], 
                X[0].permute(1, 2, 0).squeeze().detach().cpu().numpy()
            ))

        for j in range(len(Y)):
            label = Y[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
        
        targets += len(Y)
        corrects = 0
        for item in class_correct: corrects += item 
        s = ('%10.3g' * 4) % ( (100*corrects/targets), args['eps'], corrects, targets )
        pbar.set_description(s)
    
    for i in range(num_classes):
        result[ args['classes'][i] ] = 100 * class_correct[i] / class_total[i]
        result['total_acc'] += (100 * class_correct[i] / class_total[i]) / num_classes 
    
    result['adv_examples'] = adv_ex

    return result


def denormalize(imgs, args):
    for t, m, s in zip(imgs, args['normalize']['mean'], args['normalize']['std']):
        t.mul_(s).add_(m)


def save_accuracies(epsilons: list, accuracies: list, args: dict):
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 105., step=5.))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(args['folder'] + f"results_{args['attack']}.png", dpi=300, transparent=True)


def save_examples(epsilons: list, examples: list, args: dict):
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex)
    plt.tight_layout()
    plt.savefig(args['folder'] + f"examples_{args['attack']}.png", dpi=300, transparent=True)
