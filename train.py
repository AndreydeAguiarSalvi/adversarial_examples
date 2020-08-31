###################
# General classes #
###################
import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter

##############
# My classes #
##############
from utils.utils import *
from utils.datasets import get_Dataset


def train(model, args, train_loader, valid_loader, tb_writer, criterion, optimizer, scheduler):
    best_fitness = .0
    nb = len(train_loader)
    wo_best = 0
    ##########
    # Epochs #
    ##########
    t0 = time.time()
    for epoch in range(args['epochs']):
        model.train()
        mloss = torch.zeros(1)
        targets = 0
        print(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'Loss', 'Targets'))
        pbar = tqdm.tqdm(enumerate(train_loader), total=nb)  # progress bar    
        
        ################
        # Mini-batches #
        ################
        for i, (X, Y) in pbar:
            X, Y = X.to(args['device']), Y.to(args['device'])
            # model inference
            Y_hat = model(X)
            # model loss and backpropagation
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()
            # cleaning gradients
            optimizer.zero_grad()

            targets += len(Y)
            mloss = (mloss * i + loss.item()) / (i + 1)  # update mean loss
            mem = torch.cuda.memory_cached()
            s = ('%10s' + '%10.3g' * 3) % ('%g/%g' % (epoch, args['epochs'] - 1), mem, mloss, targets)
            pbar.set_description(s)

        # Update scheduler
        scheduler.step()
        # Validation loss
        validation = evaluate(model, criterion, args, valid_loader)
        
        # Saving
        torch.save(model.state_dict(), args['last'])
        if validation['total_acc'] > best_fitness:
            best_fitness = validation['total_acc']
            wo_best = 0
            torch.save(model.state_dict(), args['best'])
            f = open(args['folder'] + 'results.txt', 'w')
            for (key, value) in validation.items():
                f.write(f"{key}: {value}\n")
            f.close()
        else:
            wo_best += 1
            if wo_best == 15 and epoch > 50: 
                print('Ending training due to early stop')
                break
    
    t1 = time.time()
    print(f'{epoch} epochs completed in {(t1 - t0)/ 3600} hours.')


if __name__ == "__main__":
    args = create_argparser()
    if args['device'] != 'cpu': args['device'] = 'cuda:' + args['device']
    create_folder(args)
    
    f = open(args['folder'] + 'config.txt', 'w')
    for (key, value) in args.items():
        f.write(f"{key}: {value}\n")
    f.close()
    
    train_loader, valid_loader, _ = get_Dataset(args, True, False)
    model = create_model(args)

    args['best'] = args['folder'] + 'best.pt'
    args['last'] = args['folder'] + 'last.pt'
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args['lr0'], momentum=args['momentum'], nesterov=True, weight_decay=1E-8)
    milestones = args['epochs_decay']
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= milestones, gamma= args['gamma'])

    # tb_writer = SummaryWriter(log_dir= args['folder'] + 'runs/')

    train(model, args, train_loader, valid_loader, None, criterion, optimizer, scheduler)