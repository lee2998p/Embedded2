import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange

#imported to convert target into an nd array from list
import numpy as np
from numpy import array

from functools import reduce

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from utils.clr import CyclicLR

# def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, scheduler):
def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, scheduler):
    # model.train()
    net.train()

    correct1, correct5 = 0, 0
    print("reached train function! \n")
    for batch_idx, (data, target) in enumerate(loader): #current try: data is a tensor, target is a tensor
        print("visiting train. proceeding batch (", batch_idx, ")")
        # for i in target
        # print("type of data variable before: ", type(target), "\n")
        # print("target before: ", target, "\n")
        # target = [torch.FloatTensor(i) for i in target]
        # target = torch.FloatTensor(target)
        # print("target after: ", target, "\n")
        # print("type of data variable after: ", type(target), "\n")

        # print(target)
        # target = np.asarray(target[0])
        # print(target)
        # target = reduce(lambda x,y: x+y,target_rec) 
        # print("type of data variable is: ", type(data), "\n")
        # print("type of target variable is: ", type(target), "\n")
        # print("dimensions of target variable is: ",target.shape, "\n")
        if isinstance(scheduler, CyclicLR):
            scheduler.batch_step()
        # data, target = data.to(device=device, dtype=dtype), target.to(device=device) #device is coming as 'cuda'

        optimizer.zero_grad() #passed
        # data = data.float() #(error, possibly: partial understanding)
        output = model(data) #passed
        
        print("output from network: ", output, "\n\n")

        # loss = criterion(output, target) #current
        loss_l, loss_c = criterion(output, target)

        print(loss_l)
        print(loss_c)

        loss = loss_l + loss_c
        loss.backward()

        optimizer.step()

        corr = correct(output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Top-1 accuracy: {:.2f}%({:.2f}%). '
                'Top-5 accuracy: {:.2f}%({:.2f}%).'.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item(),
                                                           100. * corr[0] / batch_size,
                                                           100. * correct1 / (batch_size * (batch_idx + 1)),
                                                           100. * corr[1] / batch_size,
                                                           100. * correct5 / (batch_size * (batch_idx + 1))))
    return loss.item(), correct1 / len(loader.dataset), correct5 / len(loader.dataset)

# def test(model, loader, criterion, device, dtype):
def test(net, loader, criterion, device, dtype):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

    test_loss /= len(loader)

    tqdm.write(
        '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct5),
                                       len(loader.dataset), 100. * correct5 / len(loader.dataset)))
    return test_loss, correct1 / len(loader.dataset), correct5 / len(loader.dataset)


def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res

#not important for our implementation
def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)
        

def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6, max_lr=8e-5, step_size=2000,
                    mode='triangular', save_path='.'):
    model.train()
    correct1, correct5 = 0, 0
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
    epoch_count = step_size // len(loader)  # Assuming step_size is multiple of batch per epoch
    accuracy = []
    for _ in trange(epoch_count):
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if scheduler is not None:
                scheduler.batch_step()
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            corr = correct(output, target)
            accuracy.append(corr[0] / data.shape[0])

    lrs = np.linspace(min_lr, max_lr, step_size)
    plt.plot(lrs, accuracy)
    plt.show()
    plt.savefig(os.path.join(save_path, 'find_bounds_clr.png'))
    np.save(os.path.join(save_path, 'acc.npy'), accuracy)
    return
