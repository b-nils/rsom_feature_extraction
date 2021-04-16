# class for one CNN experiment

import os
import sys
import copy
import json
import warnings
from timeit import default_timer as timer

from datetime import date
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
from scipy import ndimage

from ._model import UNet, Fcn
import laynet._metrics as lfs  # lfs=lossfunctions
from ._dataset import RSOMLayerDataset, RSOMLayerDatasetUnlabeled, \
    RandomZShift, ZeroCenter, CropToEven, DropBlue, \
    ToTensor, precalcLossWeight, SwapDim, to_numpy
from utils import save_nii


class LayerNetBase():
    """
    stripped base class for predicting RSOM layers.
    for training user class LayerNet
    Args:
        device             torch.device()     'cuda' 'cpu'
        dirs               dict of string      use these directories
        filename           string              pattern to save output
    """

    def __init__(self,
                 dirs={'train': '', 'eval': '', 'pred': '', 'model': '', 'out': ''},
                 device=torch.device('cuda'),
                 model_depth=4,
                 probability=0.5,
                 model_type='unet'
                 ):

        self.model_depth = model_depth
        self.dirs = dirs
        self.out_pred_dir = dirs['out']
        self.probability = probability

        self.pred_dataset = RSOMLayerDatasetUnlabeled(
            dirs['pred'],
            transform=transforms.Compose([
                ZeroCenter(),
                CropToEven(network_depth=self.model_depth),
                DropBlue(),
                ToTensor()])
        )

        self.pred_dataloader = DataLoader(
            self.pred_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        self.size_pred = len(self.pred_dataset)
        self.device = device
        self.dtype = torch.float32

        if model_type == 'unet':
            self.model = UNet(in_channels=2,
                              n_classes=1,
                              depth=model_depth,
                              wf=6,
                              padding=True,
                              batch_norm=True,
                              up_mode='upconv',
                              dropout=True)
        elif model_type == 'fcn':
            self.model = Fcn()
        else:
            raise NotImplementedError
        self.model = self.model.to(self.device)

        self.minibatch_size = 1 if model_type == 'unet' else 9

        if self.dirs['model']:
            print('Loading LayerNet model from:', self.dirs['model'])
            self.model.load_state_dict(torch.load(self.dirs['model']))

    def printandlog(self, *msg):
        if 1:
            print(*msg)
            try:
                print(*msg, file=self.logfile)
            except:
                pass

    def calc_metrics(self, ground_truth, label):
        print(ground_truth.shape)
        prec = []
        recall = []
        dice = []
        for i in range(ground_truth.shape[1]):
            prec.append(lfs.calc_precision(ground_truth[:, i, ...], label[:, i, ...]))
            recall.append(lfs.calc_recall(ground_truth[:, i, ...], label[:, i, ...]))
            dice.append(lfs.calc_dice(ground_truth[:, i, ...], label[:, i, ...]))
        return prec, recall, dice

    def predict(self):
        self.model.eval()
        iterator = iter(self.pred_dataloader)

        for i in range(self.size_pred):
            # get the next volume to evaluate
            batch = next(iterator)

            m = batch['meta']

            batch['data'] = batch['data'].to(
                self.device,
                self.dtype,
                non_blocking=True
            )

            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                                    step=self.minibatch_size)
            # init empty prediction stack
            shp = batch['data'].shape
            # [0 x 2 x 500 x 332]
            prediction_stack = torch.zeros((0, 1, shp[3], shp[4]),
                                           dtype=self.dtype,
                                           requires_grad=False
                                           )
            prediction_stack = prediction_stack.to(self.device)

            for i2, idx in enumerate(minibatches):
                if idx + self.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:, idx:idx + self.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]

                data = torch.squeeze(data, dim=0)

                prediction = self.model(data)

                prediction = prediction.detach()
                prediction_stack = torch.cat((prediction_stack, prediction), dim=0)

            prediction_stack = prediction_stack.to('cpu')

            # transform -> labels
            prediction_stack = torch.sigmoid(prediction_stack)

            label = prediction_stack >= self.probability

            m = batch['meta']

            label = label.squeeze()
            label = to_numpy(label, m)

            filename = batch['meta']['filename'][0]
            filename = filename.replace('rgb.nii.gz', '')

            if 1:
                label = self.smooth_pred(label, filename)

            # print('Saving', filename)
            save_nii(label.astype(np.uint8), self.out_pred_dir, filename + 'pred')

            if 0:
                save_nii(to_numpy(prediction_stack.squeeze(), m),
                         self.out_pred_dir,
                         filename + 'ppred')
            # compare to ground truth
            if 0:
                label_gt = batch['label']

                label_gt = torch.squeeze(label_gt, dim=0)
                label_gt = to_numpy(label_gt, m)

                label_diff = (label > label_gt).astype(np.uint8)
                label_diff += 2 * (label < label_gt).astype(np.uint8)
                # label_diff = label != label_gt
                save_nii(label_diff, self.out_pred_dir, filename + 'dpred')

    def predict_calc(self):
        self.model.eval()
        iterator = iter(self.pred_dataloader)
        prec = []
        recall = []
        dice = []
        for i in range(self.size_pred):
            # get the next volume to evaluate
            batch = next(iterator)

            m = batch['meta']

            batch['data'] = batch['data'].to(
                self.device,
                self.dtype,
                non_blocking=True
            )

            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                                    step=self.minibatch_size)
            # init empty prediction stack
            shp = batch['data'].shape
            # [0 x 2 x 500 x 332]
            prediction_stack = torch.zeros((0, 1, shp[3], shp[4]),
                                           dtype=self.dtype,
                                           requires_grad=False
                                           )
            prediction_stack = prediction_stack.to(self.device)

            for i2, idx in enumerate(minibatches):
                if idx + self.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:, idx:idx + self.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]

                data = torch.squeeze(data, dim=0)

                prediction = self.model(data)

                prediction = prediction.detach()
                prediction_stack = torch.cat((prediction_stack, prediction), dim=0)

            prediction_stack = prediction_stack.to('cpu')

            # transform -> labels
            prediction_stack = torch.sigmoid(prediction_stack)

            label = prediction_stack >= self.probability

            m = batch['meta']

            label = label.squeeze()
            label = to_numpy(label, m)

            print('in pred: max label', batch['label'].max())
            ground_truth = batch['label']
            print('gt shape', ground_truth.shape)
            ground_truth = ground_truth.squeeze()
            ground_truth = to_numpy(ground_truth, m)

            assert label.shape == ground_truth.shape
            p, r, d = self.calc_metrics(ground_truth=ground_truth, label=label)
            prec.append(p)
            recall.append(r)
            dice.append(d)

            filename = batch['meta']['filename'][0]
            filename = filename.replace('rgb.nii.gz', '')

            if 1:
                label = self.smooth_pred(label, filename)

            # print('Saving', filename)
            if 1:
                save_nii(label.astype(np.uint8), self.out_pred_dir, filename + 'pred')

            if 0:
                save_nii(to_numpy(prediction_stack.squeeze(), m),
                         self.out_pred_dir,
                         filename + 'ppred')
            # compare to ground truth
            if 0:
                label_gt = batch['label']

                label_gt = torch.squeeze(label_gt, dim=0)
                label_gt = to_numpy(label_gt, m)

                label_diff = (label > label_gt).astype(np.uint8)
                label_diff += 2 * (label < label_gt).astype(np.uint8)
                # label_diff = label != label_gt
                save_nii(label_diff, self.out_pred_dir, filename + 'dpred')

        self.printandlog('Metrics:')
        self.printandlog('Precision: mean {:.5f} std {:.5f}'.format(np.nanmean(prec), np.nanstd(prec)))
        self.printandlog('Recall:    mean {:.5f} std {:.5f}'.format(np.nanmean(recall), np.nanstd(recall)))
        self.printandlog('Dice:      mean {:.5f} std {:.5f}'.format(np.nanmean(dice), np.nanstd(dice)))

    @staticmethod
    def smooth_pred(label, filename):
        '''
        smooth the prediction
        '''

        # for every slice in x-y plane, calculate label sum
        label_sum = np.sum(label, axis=(1, 2))

        max_occupation = np.amax(label_sum) / (label.shape[1] * label.shape[2])

        # print('Max occ', max_occupation)
        # print('idx max occ', max_occupation_idx)
        if max_occupation >= 0.01:
            # normalize
            label_sum = label_sum.astype(np.double) / np.amax(label_sum)

            # define cutoff parameter
            cutoff = 0.05

            label_sum_bin = label_sum > cutoff

            label_sum_idx = np.squeeze(np.nonzero(label_sum_bin))

            layer_end = label_sum_idx[-1]
        else:
            print("not working!!")

        label[layer_end:, :, :] = 0

        # 1. fill holes inside the label
        ldtype = label.dtype
        label = ndimage.binary_fill_holes(label).astype(ldtype)
        label_shape = label.shape
        label = np.pad(label, 2, mode='edge')
        label = ndimage.binary_closing(label, iterations=2)
        label = label[2:-2, 2:-2, 2:-2]
        assert label_shape == label.shape

        # 2. scan along z-dimension change in label 0->1 1->0
        #    if there's more than one transition each, one needs to be dropped
        #    after filling holes, we hope to be able to drop the outer one
        # 3. get 2x 2-D surface data with surface height being the index in z-direction

        surf_lo = np.zeros((label_shape[1], label_shape[2]))

        # set highest value possible (500) as default. Therefore, empty sections
        # of surf_up and surf_lo will get smoothened towards each other, and during
        # reconstructions, we won't have any weird shapes.
        surf_up = surf_lo.copy() + label_shape[0]

        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):
                nz = np.nonzero(label[:, xx, yy])

                if nz[0].size != 0:
                    idx_up = nz[0][0]
                    idx_lo = nz[0][-1]
                    surf_up[xx, yy] = idx_up
                    surf_lo[xx, yy] = idx_lo

        #    smooth coarse structure, eg with a 25x25 average and crop everything which is above average*factor
        #           -> hopefully spikes will be removed.
        surf_up_m = ndimage.median_filter(surf_up, size=(26, 26), mode='nearest')
        surf_lo_m = ndimage.median_filter(surf_lo, size=(26, 26), mode='nearest')

        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):
                if surf_up[xx, yy] < surf_up_m[xx, yy]:
                    surf_up[xx, yy] = surf_up_m[xx, yy]
                if surf_lo[xx, yy] > surf_lo_m[xx, yy]:
                    surf_lo[xx, yy] = surf_lo_m[xx, yy]

        # apply suitable kernel in order to smooth
        # smooth fine structure, eg with a 5x5 moving average
        surf_up = ndimage.uniform_filter(surf_up, size=(9, 5), mode='nearest')
        surf_lo = ndimage.uniform_filter(surf_lo, size=(9, 5), mode='nearest')

        # 5. reconstruct label
        label_rec = np.zeros(label_shape, dtype=np.uint8)
        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):
                label_rec[int(np.round(surf_up[xx, yy])):int(np.round(surf_lo[xx, yy])), xx, yy] = 1

        return label_rec


class LayerNet(LayerNetBase):
    '''
    class for setting up, training and evaluating of layer segmentation
    with unet on RSOM dataset
    Args:
        device             torch.device()     'cuda' 'cpu'
        model_depth        int                 unet depth
        dataset_zshift     int or (int, int)   data aug. zshift
        dirs               dict of string      use these directories
        filename           string              pattern to save output
        optimizer          string
        initial_lr         float               initial learning rate
        scheduler_patience int                 n epochs before lr reduction
        lossfn             function            custom lossfunction
        class_weight       (float, float)      class weight for classes (0, 1)
        epochs             int                 number of epochs
    '''

    def __init__(self,
                 device=torch.device('cuda'),
                 sdesc='',
                 model_depth=3,
                 model_type='unet',
                 dataset_zshift=0,
                 dirs={'train': '', 'eval': '', 'model': '', 'pred': '', 'out': ''},
                 optimizer='Adam',
                 initial_lr=1e-4,
                 scheduler_patience=3,
                 lossfn=lfs.custom_loss_1,
                 lossfn_smoothness=0,
                 lossfn_window=5,
                 lossfn_spatial_weight_scale=True,
                 class_weight=None,
                 epochs=30,
                 dropout=False,
                 DEBUG=False,
                 probability=0.5,
                 slice_wise=False
                 ):
        self.slice_wise = slice_wise
        if not slice_wise:
            self.eval_batch_size = 1
            self.train_batch_size = 1
        else:
            self.eval_batch_size = 5
            self.train_batch_size = 5

        self.sdesc = sdesc

        self.DEBUG = DEBUG
        self.LOG = True
        if DEBUG:
            print('DEBUG MODE')
        #
        out_root_list = os.listdir(dirs['out'])

        today = date.today().strftime('%y%m%d')
        today_existing = [el for el in out_root_list if today in el]
        if today_existing:
            nr = max([int(el[7:9]) for el in today_existing]) + 1
        else:
            nr = 0

        self.dirs = dirs

        self.today_id = today + '-{:02d}'.format(nr)
        self.dirs['out'] = os.path.join(self.dirs['out'], self.today_id)
        if self.sdesc:
            self.dirs['out'] += '-' + self.sdesc

        self.debug('Output directory string:', self.dirs['out'])

        # PROCESS LOGGING
        if not self.DEBUG:
            os.mkdir(self.dirs['out'])
            if self.LOG:
                try:
                    self.logfile = open(os.path.join(self.dirs['out'],
                                                     'log' + self.today_id), 'x')
                except:
                    print('Couldn\'n open logfile')
            else:
                self.logfile = None

        if self.dirs['pred']:
            if not self.DEBUG:
                self.out_pred_dir = os.path.join(self.dirs['out'], 'prediction')
                os.mkdir(self.out_pred_dir)
        # MODEL
        if model_type == 'unet':
            self.model = UNet(in_channels=2,
                              n_classes=1,
                              depth=model_depth,
                              wf=6,
                              padding=True,
                              batch_norm=True,
                              up_mode='upconv',
                              dropout=dropout)
        elif model_type == 'fcn':
            self.model = Fcn()
        else:
            raise NotImplementedError
        self.model_dropout = dropout

        self.model = self.model.to(device)
        self.model = self.model.float()

        if model_type == 'unet':
            print(self.model.down_path[0].block.state_dict()['0.weight'].device)

        self.model_depth = model_depth

        # LOSSFUNCTION
        self.lossfn = lossfn
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
            self.class_weight = self.class_weight.to(device)
        else:
            self.class_weight = None

        self.lossfn_smoothness = lossfn_smoothness
        self.lossfn_window = lossfn_window
        self.lossfn_spatial_weight_scale = lossfn_spatial_weight_scale

        # DATASET
        self.train_dataset_zshift = dataset_zshift

        self.train_dataset = RSOMLayerDataset(self.dirs['train'],
                                              slice_wise=self.slice_wise,
                                              transform=transforms.Compose([RandomZShift(dataset_zshift),
                                                                            ZeroCenter(),
                                                                            CropToEven(network_depth=self.model_depth),
                                                                            DropBlue(),
                                                                            ToTensor(shuffle=True),
                                                                            precalcLossWeight()
                                                                            ]))

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.train_batch_size,
                                           shuffle=True,
                                           drop_last=False,
                                           num_workers=4,
                                           pin_memory=True)

        self.eval_dataset = RSOMLayerDataset(self.dirs['eval'],
                                             slice_wise=self.slice_wise,
                                             transform=transforms.Compose([RandomZShift(),
                                                                           ZeroCenter(),
                                                                           CropToEven(network_depth=self.model_depth),
                                                                           DropBlue(),
                                                                           ToTensor(),
                                                                           precalcLossWeight()]))
        self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=self.eval_batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=4,
                                          pin_memory=True)
        if dirs['pred']:
            self.pred_dataset = RSOMLayerDataset(
                dirs['pred'],
                transform=transforms.Compose([
                    ZeroCenter(),
                    CropToEven(network_depth=self.model_depth),
                    DropBlue(),
                    ToTensor()])
            )

            self.pred_dataloader = DataLoader(self.pred_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

        self.probability = probability

        # OPTIMIZER
        self.initial_lr = initial_lr
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.initial_lr,
                weight_decay=0
            )

        # SCHEDULER
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=scheduler_patience,
            verbose=True,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-8)

        # HISTORY
        self.history = {
            'train': {'epoch': [], 'loss': []},
            'eval': {'epoch': [], 'loss': []}
        }

        # CURRENT EPOCH
        self.curr_epoch = None

        # ADDITIONAL ARGS
        self.args = self.helperClass()

        self.size_pred = len(self.pred_dataset)
        self.args.size_train = len(self.train_dataset)
        self.args.size_eval = len(self.eval_dataset)
        self.args.minibatch_size = 5 if model_type == 'unet' else 9
        self.minibatch_size = self.args.minibatch_size
        self.args.device = device
        self.device = device
        self.args.dtype = torch.float32
        self.dtype = self.args.dtype
        self.args.non_blocking = True
        self.args.n_epochs = epochs
        self.args.data_dim = self.eval_dataset[0]['data'].shape

    def debug(self, *msg):
        if self.DEBUG:
            print(*msg)

    def printConfiguration(self, destination='stdout'):
        if destination == 'stdout':
            where = sys.stdout
        elif destination == 'logfile' and not self.DEBUG:
            where = self.logfile

        if destination == 'logfile' and self.DEBUG:
            pass
        else:
            print('LayerUNET configuration:', file=where)
            print('DATA: train dataset loc:', self.dirs['train'], file=where)
            print('      train dataset len:', self.args.size_train, file=where)
            print('      eval dataset loc:', self.dirs['eval'], file=where)
            print('      eval dataset len:', self.args.size_eval, file=where)
            print('      shape:', self.args.data_dim, file=where)
            print('      zshift:', self.train_dataset_zshift)
            print('EPOCHS:', self.args.n_epochs, file=where)
            print('OPTIMIZER:', self.optimizer, file=where)
            print('initial lr:', self.initial_lr, file=where)
            print('LOSS: fn', self.lossfn, file=where)
            print('      class_weight', self.class_weight, file=where)
            print('      smoothnes param', self.lossfn_smoothness, file=where)
            print('      window', self.lossfn_window, file=where)
            print('CNN:  unet', file=where)
            print('      depth', self.model_depth, file=where)
            print('      dropout?', self.model_dropout, file=where)
            print('OUT:  model:', self.dirs['model'], file=where)
            print('      pred:', self.dirs['pred'], file=where)
            print('')
            print(self.model, file=where)

    def train_all_epochs(self):
        self.best_model = copy.deepcopy(self.model.state_dict())
        for k, v in self.best_model.items():
            self.best_model[k] = v.to('cpu')

        self.best_loss = float('inf')

        print('Entering training loop..')
        for curr_epoch in range(self.args.n_epochs):
            # in every epoch, generate iterators
            train_iterator = iter(self.train_dataloader)

            eval_iterator = iter(self.eval_dataloader)

            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            if curr_epoch == 1:
                tic = timer()
            self.debug('train')
            self.train(iterator=train_iterator, epoch=curr_epoch)

            torch.cuda.empty_cache()
            if curr_epoch == 1:
                toc = timer()
                print('Training took:', toc - tic)
                tic = timer()
            self.debug('eval')
            self.eval(iterator=eval_iterator, epoch=curr_epoch)

            # torch.cuda.empty_cache()
            if curr_epoch == 1:
                toc = timer()
                print('Evaluation took:', toc - tic)

            print(torch.cuda.memory_cached() * 1e-6, 'MB memory used')
            # extract the average training loss of the epoch
            le_idx = self.history['train']['epoch'].index(curr_epoch)
            le_losses = self.history['train']['loss'][le_idx:]
            # divide by batch size (170) times dataset size
            if not self.slice_wise:
                train_loss = sum(le_losses) / (self.args.data_dim[0] * self.args.size_train)
            else:
                train_loss = sum(le_losses) / (self.args.size_train)
            # extract most recent eval loss
            curr_loss = self.history['eval']['loss'][-1]

            # use ReduceLROnPlateau scheduler
            self.scheduler.step(curr_loss)

            if curr_loss < self.best_loss:
                self.best_loss = copy.deepcopy(curr_loss)
                self.best_model = copy.deepcopy(self.model.state_dict())
                for k, v in self.best_model.items():
                    self.best_model[k] = v.to('cpu')
                found_nb = 'new best!'
            else:
                found_nb = ''

            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch + 1,
                self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb)
            if not self.DEBUG:
                print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                    curr_epoch + 1,
                    self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb, file=self.logfile)

        print('finished training.')

    def train(self, iterator, epoch):
        '''
        '''
        # PARSE
        # model = self.model
        # optimizer = self.optimizer
        # history = self.history
        # lossfn = self.lossfn
        # args = self.args

        self.model.train()

        for i in range(int(np.ceil(self.args.size_train / self.train_batch_size))):
            # get the next batch of training data
            try:
                batch = next(iterator)
            except StopIteration:
                print('Iterators wrong')
                break

            batch['label'] = batch['label'].to(
                self.args.device,
                dtype=self.args.dtype,
                non_blocking=self.args.non_blocking)
            batch['data'] = batch['data'].to(
                self.args.device,
                self.args.dtype,
                non_blocking=self.args.non_blocking)
            batch['meta']['weight'] = batch['meta']['weight'].to(
                self.args.device,
                self.args.dtype,
                non_blocking=self.args.non_blocking)

            if not self.slice_wise:
                # divide into minibatches
                minibatches = np.arange(batch['data'].shape[1],
                                        step=self.args.minibatch_size)
                for i2, idx in enumerate(minibatches):
                    if idx + self.args.minibatch_size < batch['data'].shape[1]:
                        data = batch['data'][:,
                               idx:idx + self.args.minibatch_size, :, :]
                        label = batch['label'][:,
                                idx:idx + self.args.minibatch_size, :, :]
                        weight = batch['meta']['weight'][:,
                                 idx:idx + self.args.minibatch_size, :, :]
                    else:
                        data = batch['data'][:, idx:, :, :]
                        label = batch['label'][:, idx:, :, :]
                        weight = batch['meta']['weight'][:, idx:, :, :]

                    data = torch.squeeze(data, dim=0)
                    label = torch.squeeze(label, dim=0)
                    weight = torch.squeeze(weight, dim=0)

                    label = torch.unsqueeze(label, dim=1)
                    weight = torch.unsqueeze(weight, dim=1)

                    prediction = self.model(data)

                    # move back to save memory
                    # prediction = prediction.to('cpu')
                    loss = self.lossfn(
                        pred=prediction,
                        target=label,
                        spatial_weight=weight,
                        class_weight=self.class_weight,
                        smoothness_weight=self.lossfn_smoothness,
                        window=self.lossfn_window,
                        spatial_weight_scale=self.lossfn_spatial_weight_scale)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    frac_epoch = epoch + \
                                 i / self.args.size_train + \
                                 i2 / (self.args.size_train * minibatches.size)

                    self.history['train']['epoch'].append(frac_epoch)
                    self.history['train']['loss'].append(loss.data.item())

            else:

                data = batch['data']
                label = batch['label']
                weight = batch['meta']['weight']

                data = torch.squeeze(data, dim=1)

                # self.debug('DATA shape', data.shape)
                # self.debug('LABEL shape', label.shape)

                prediction = self.model(data)

                loss = self.lossfn(
                    pred=prediction,
                    target=label,
                    spatial_weight=weight,
                    class_weight=self.class_weight,
                    smoothness_weight=self.lossfn_smoothness,
                    window=self.lossfn_window,
                    spatial_weight_scale=self.lossfn_spatial_weight_scale)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_loss = loss.data.item()

                frac_epoch = epoch + i / int(np.ceil(self.args.size_train / self.train_batch_size))
                self.debug('train', i, 'current loss', current_loss, 'batch', data.shape[0])
                self.history['train']['epoch'].append(frac_epoch)
                self.history['train']['loss'].append(current_loss)

    def eval(self, iterator, epoch):
        '''
        '''
        # PARSE
        # model = self.model
        # history = self.history
        # lossfn = self.lossfn
        # args = self.args

        self.model.eval()
        running_loss = 0.0

        for i in range(int(np.ceil(self.args.size_train / self.eval_batch_size))):
            # get the next batch of the testset
            try:
                batch = next(iterator)
            except StopIteration:
                print('Iterators wrong')
                break

            batch['label'] = batch['label'].to(
                self.args.device,
                dtype=self.args.dtype,
                non_blocking=self.args.non_blocking)
            batch['data'] = batch['data'].to(
                self.args.device,
                self.args.dtype,
                non_blocking=self.args.non_blocking)
            batch['meta']['weight'] = batch['meta']['weight'].to(
                self.args.device,
                self.args.dtype,
                non_blocking=self.args.non_blocking)

            if not self.slice_wise:
                # divide into minibatches
                minibatches = np.arange(batch['data'].shape[1],
                                        step=self.args.minibatch_size)
                for i2, idx in enumerate(minibatches):
                    if idx + self.args.minibatch_size < batch['data'].shape[1]:
                        data = batch['data'][:,
                               idx:idx + self.args.minibatch_size, :, :]
                        label = batch['label'][:,
                                idx:idx + self.args.minibatch_size, :, :]
                        weight = batch['meta']['weight'][:,
                                 idx:idx + self.args.minibatch_size, :, :]
                    else:
                        data = batch['data'][:, idx:, :, :]
                        label = batch['label'][:, idx:, :, :]
                        weight = batch['meta']['weight'][:, idx:, :, :]

                    data = torch.squeeze(data, dim=0)
                    label = torch.squeeze(label, dim=0)
                    weight = torch.squeeze(weight, dim=0)

                    label = torch.unsqueeze(label, dim=1)
                    weight = torch.unsqueeze(weight, dim=1)

                    prediction = self.model(data)
                    # prediction = prediction.to('cpu')

                    loss = self.lossfn(
                        pred=prediction,
                        target=label,
                        spatial_weight=weight,
                        class_weight=self.class_weight,
                        smoothness_weight=self.lossfn_smoothness,
                        window=self.lossfn_window,
                        spatial_weight_scale=self.lossfn_spatial_weight_scale)
                    # loss running variable
                    # add value for every minibatch
                    # this should scale linearly with minibatch size
                    # have to verify!
                    running_loss += loss.data.item()


            else:
                data = batch['data']
                label = batch['label']
                weight = batch['meta']['weight']

                data = torch.squeeze(data, dim=1)

                prediction = self.model(data)

                loss = self.lossfn(
                    pred=prediction,
                    target=label,
                    spatial_weight=weight,
                    class_weight=self.class_weight,
                    smoothness_weight=self.lossfn_smoothness,
                    window=self.lossfn_window,
                    spatial_weight_scale=self.lossfn_spatial_weight_scale)

                # loss running variable
                # TODO: check if this works
                # add value for every minibatch
                # this should scale linearly with minibatch size
                # have to verify!
                current_loss = loss.data.item()

                running_loss += current_loss
                self.debug('eval', i, 'current loss', current_loss, 'batch', data.shape[0])

        if self.slice_wise:
            epoch_loss = running_loss / (self.args.size_eval)
        else:
            epoch_loss = running_loss / (self.args.size_eval * batch['data'].shape[1])

        self.history['eval']['epoch'].append(epoch)
        self.history['eval']['loss'].append(epoch_loss)

    def predict_calc(self):
        self.model.eval()
        iterator = iter(self.pred_dataloader)
        prec = []
        recall = []
        dice = []
        for i in range(self.size_pred):
            # get the next volume to evaluate
            batch = next(iterator)

            m = batch['meta']

            batch['data'] = batch['data'].to(
                self.device,
                self.dtype,
                non_blocking=True
            )

            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                                    step=self.minibatch_size)
            # init empty prediction stack
            shp = batch['data'].shape
            # [0 x 2 x 500 x 332]
            prediction_stack = torch.zeros((0, 1, shp[3], shp[4]),
                                           dtype=self.dtype,
                                           requires_grad=False
                                           )
            prediction_stack = prediction_stack.to(self.device)

            for i2, idx in enumerate(minibatches):
                if idx + self.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:, idx:idx + self.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]

                data = torch.squeeze(data, dim=0)

                prediction = self.model(data)

                prediction = prediction.detach()
                prediction_stack = torch.cat((prediction_stack, prediction), dim=0)

            prediction_stack = prediction_stack.to('cpu')

            # transform -> labels
            prediction_stack = torch.sigmoid(prediction_stack)

            label = prediction_stack >= self.probability

            m = batch['meta']

            label = label.squeeze()
            label = to_numpy(label, m)

            print('in pred: max label', batch['label'].max())
            ground_truth = batch['label']
            print('gt shape', ground_truth.shape)
            ground_truth = ground_truth.squeeze()
            ground_truth = to_numpy(ground_truth, m)

            assert label.shape == ground_truth.shape
            p, r, d = self.calc_metrics(ground_truth=ground_truth, label=label)
            prec.append(p)
            recall.append(r)
            dice.append(d)

            filename = batch['meta']['filename'][0]
            filename = filename.replace('rgb.nii.gz', '')

            if 0:
                label = self.smooth_pred(label, filename)

            # print('Saving', filename)
            if not self.DEBUG:
                save_nii(label.astype(np.uint8), self.out_pred_dir, filename + 'pred')

            if not self.DEBUG:
                save_nii(to_numpy(prediction_stack.squeeze(), m),
                         self.out_pred_dir,
                         filename + 'ppred')
            # compare to ground truth
            if 0:
                label_gt = batch['label']

                label_gt = torch.squeeze(label_gt, dim=0)
                label_gt = to_numpy(label_gt, m)

                label_diff = (label > label_gt).astype(np.uint8)
                label_diff += 2 * (label < label_gt).astype(np.uint8)
                # label_diff = label != label_gt
                save_nii(label_diff, self.out_pred_dir, filename + 'dpred')

        self.printandlog('Metrics:')
        self.printandlog('Precision: mean {:.5f} std {:.5f}'.format(np.nanmean(prec), np.nanstd(prec)))
        self.printandlog('Recall:    mean {:.5f} std {:.5f}'.format(np.nanmean(recall), np.nanstd(recall)))
        self.printandlog('Dice:      mean {:.5f} std {:.5f}'.format(np.nanmean(dice), np.nanstd(dice)))

    def save_code_status(self):
        if not self.DEBUG:
            try:
                path = os.path.join(self.dirs['out'], 'git')
                os.system('git log -1 | head -n 1 > {:s}.diff'.format(path))
                os.system('echo /"\n/" >> {:s}.diff'.format(path))
                os.system('git diff >> {:s}.diff'.format(path))
            except:
                self.printandlog('Saving git diff FAILED!')

    def calc_metrics(self, ground_truth, label):
        print(ground_truth.shape)
        prec = []
        recall = []
        dice = []
        for i in range(ground_truth.shape[1]):
            prec.append(lfs.calc_precision(ground_truth[:, i, ...], label[:, i, ...]))
            recall.append(lfs.calc_recall(ground_truth[:, i, ...], label[:, i, ...]))
            dice.append(lfs.calc_dice(ground_truth[:, i, ...], label[:, i, ...]))
        return prec, recall, dice

    def save_model(self, model='best', pat=''):
        if not self.DEBUG:
            if model == 'best':
                save_this = self.best_model
            elif model == 'last':
                save_this = self.last_model

            torch.save(save_this, os.path.join(self.dirs['out'], 'mod' + self.today_id + pat + '.pt'))

            json_f = json.dumps(self.history)
            f = open(os.path.join(self.dirs['out'], 'hist_' + self.today_id + pat + '.json'), 'w')
            f.write(json_f)
            f.close()

    class helperClass():
        pass
