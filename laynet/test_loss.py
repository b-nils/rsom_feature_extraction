import torch
import torch.nn as nn
import numpy as np




# TEST NEW LOSSFUNCTIONS

# output from CNN has shape
# [Batchsize x nClasses x d...]



# nn.CrossEntropyLoss()
# wants shape [Batchsize x d..]
# with entries 0, 1, 2 ... for classes
from _metrics import smoothness_loss, smoothness_loss_new

# create fake inputs

pred = np.zeros((2, 2, 10, 10))

# generate  for class 1:
pred_ideal = pred.copy()
pred_ideal[:, 1, 5, :] = 1

# INSERT MODIFICATION (nonideal)
pred_ideal[:, 1, 0:1,:] = 1
pred_ideal[:, 1, 6, 6] = 1 
pred_ideal[:, 0, :, :] = np.logical_not(pred_ideal[:, 1, :, :])
pred_ideal = torch.from_numpy(pred_ideal).to('cuda')

# print(pred_ideal.shape)
# print(pred_ideal)

smoothness_loss(pred_ideal)

def smtest(pred, window=5):
    print(pred)
    pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).cuda().float()
    print('COST:', smoothness_loss_new(pred,window=window))

# NEW TESTS
pred = np.zeros((7, 7))
pred[0,:] = 1
smtest(pred)

pred = np.zeros((7, 7))
pred[:,:] = 1
smtest(pred)

pred = np.zeros((7, 7))
pred[:,4] = 1
smtest(pred)

# more batches
pred = np.zeros((2, 1, 7, 7))
pred[0,0,:,4] = 1
print(pred)
pred = torch.from_numpy(pred).cuda().float()
print('COST:', smoothness_loss_new(pred))

pred = np.zeros((2, 1, 7, 7))
pred[0,0,2,:] = 1
pred[0,0,3,:] = 1
print(pred)
pred = torch.from_numpy(pred).cuda().float()
print('COST:', smoothness_loss_new(pred))

# NEW TESTS
for win in [3, 7]:
    print('window =', win)
    pred = np.zeros((7, 7))
    pred[0,:] = 1
    smtest(pred, window=win)

    pred = np.zeros((7, 7))
    pred[:,:] = 1
    smtest(pred,window=win)

    pred = np.zeros((7, 7))
    pred[3,4] = 0.8
    pred[3,3] = 0.5
    smtest(pred,window=win)


