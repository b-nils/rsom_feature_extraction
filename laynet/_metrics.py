import torch
import warnings
# from torch import nn
import copy 
import numpy as np
import math

def custom_loss_1(pred, target, spatial_weight, class_weight=None):
    '''
    doc string
    '''
    fn = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='none')
    
    pred = pred.float()
    target = target.long()
    unred_loss = fn(pred, target)

    loss = spatial_weight.float() * unred_loss
    loss = torch.sum(loss)
    
    return loss 
 
def custom_loss_1_smooth(pred, target, spatial_weight, class_weight=None, smoothness_weight=100):

    fn = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='none')
    
    pred = pred.float()
    target = target.long()
    unred_loss = fn(pred, target)

    loss = spatial_weight.float() * unred_loss
    loss = torch.sum(loss)
    
    # add smoothness loss
    more =  smoothness_weight*smoothness_loss(pred)
    
    print('H_', loss, 'S_', more)
    loss += more
    
    return loss 

def bce_and_smooth(pred, 
                  target, 
                  spatial_weight, 
                  class_weight=None, 
                  smoothness_weight=100, 
                  spatial_weight_scale=True,
                  window=5):
    
    # CROSS ENTROPY PART
    f_H = torch.nn.BCEWithLogitsLoss(reduction='none',
                                     pos_weight=class_weight)

    # print('Pred', pred.shape)
    # print('Target', target.shape)
    # print('spatial_weight', spatial_weight.shape)
    H = f_H(pred, target)

    # scale with spatial weight
    if spatial_weight_scale:
        H = spatial_weight.float() * H

    H = torch.sum(H)

    # SMOOTHNESS PART
    f_S = torch.nn.Sigmoid()

    S = f_S(pred)

    S = smoothness_weight * smoothness_loss_new(S, window=window)
    # print('H', H, 'S', S)
    return H + S

def smoothness_loss_new(S, window=5):
    
    # print('S minmax', S.min(), S.max())
    # print('S.shape', S.shape) 
    
    pred_shape = S.shape 
    
    S = S.view(-1)
   
    # add 2 extra dimensions
    # conv1d needs input of shape
    # [minibatch x in_channels x iW]
    label = torch.unsqueeze(S, 0)
    label = torch.unsqueeze(label, 0)

    # weights of the convolutions are simply 1, and divided by the window size
    weight = torch.ones(1, 1, window).float().to('cuda') / window

    label_conv = torch.nn.functional.conv1d(input=label, 
            weight=weight,
            padding=int(math.floor(window/2)))
    
    label_conv = torch.squeeze(label_conv)
    label = torch.squeeze(label)
    
    # for perfectly smooth label, this value is zero
    # e.g. if label_conv[i] = label[i], -> 1/1 - 1 = 0
    label_smoothness =torch.abs(((label_conv + 1) / (label + 1)) - 1)
    
    # edge correction, steps at the boundaries do not count as unsmooth,
    # therefore corresponding entries of label_smoothness are zeroed out
    edge_corr = torch.zeros((pred_shape[3])).to('cuda')
    edge_corr[int(math.floor(window/2)):-int(math.floor(window/2))] = 1
    edge_corr = edge_corr.repeat(pred_shape[0]*pred_shape[2])

    label_smoothness *= edge_corr
   
    # print('after smooth', label_smoothness.shape)
    # print('min max', label_smoothness.min(), label_smoothness.max())
    # target shape
    # [minibatch x Z x X]
    
    # return some loss measure, as the sum of all smoothness losses
    return torch.sum(label_smoothness)    
    
def smoothness_loss(pred, window=5):
    '''
    smoothness loss x-y plane, ie. perfect label
    separation in z-direction will cause zero loss
    
    first try only calculating the loss on label "1"
    as this is a 2-label problem only anyways
    '''
    pred_shape = pred.shape 
   
    # this gives nonzero entries for label "1"
    label = (pred[:,1,:,:] - pred[:,0,:,:]).float()
    label = torch.nn.functional.relu(label)

    label = label.view(-1)

    # add 2 extra dimensions
    # conv1d needs input of shape
    # [minibatch x in_channels x iW]
    label = torch.unsqueeze(label, 0)
    label = torch.unsqueeze(label, 0)

    # weights of the convolutions are simply 1, and divided by the window size
    weight = torch.ones(1, 1, window).float().to('cuda') / window

    label_conv = torch.nn.functional.conv1d(input=label, 
            weight=weight,
            padding=int(math.floor(window/2)))
    
    label_conv = torch.squeeze(label_conv)
    label = torch.squeeze(label)
    
    # for perfectly smooth label, this value is zero
    # e.g. if label_conv[i] = label[i], -> 1/1 - 1 = 0
    label_smoothness =torch.abs((label_conv+1) / (label+1)-1)
    
    # edge correction, steps at the boundaries do not count as unsmooth,
    # therefore corresponding entries of label_smoothness are zeroed out
    edge_corr = torch.zeros((pred_shape[3])).to('cuda')
    edge_corr[int(math.floor(window/2)):-int(math.floor(window/2))] = 1
    edge_corr = edge_corr.repeat(pred_shape[0]*pred_shape[2])

    label_smoothness *= edge_corr
    
    # target shape
    # [minibatch x Z x X]
    
    # return some loss measure, as the sum of all smoothness losses
    return torch.sum(label_smoothness)    


def calc_recall(label, pred):
    label = label.astype(np.bool)
    pred = pred.astype(np.bool)
    TP = np.sum(np.logical_and(label, pred))
    FN = np.sum(np.logical_and(label, np.logical_not(pred)))
    
    R = TP / (TP + FN)
    return R

def calc_precision(label, pred):
    label = label.astype(np.bool)
    pred = pred.astype(np.bool)
    TP = np.sum(np.logical_and(label, pred))
    FP = np.sum(np.logical_and(pred, np.logical_not(label)))
    
    P = TP / (TP + FP) 
    return P

def calc_dice(a, b):
    return _dice(a,b)


def _dice(x, y):
    '''
    do the test in numpy
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = x.astype(np.bool)
    y = y.astype(np.bool)

    i = np.logical_and(x,y)

    if x.sum() + y.sum() == 0:
        print('Dice No True values!')
        return 1.

    return (2. * i.sum()) / (x.sum() + y.sum())




















