# class for one CNN experiment

import os
import sys
import copy
import json
import warnings
from timeit import default_timer as timer
import nibabel as nib
import concurrent.futures

from types import SimpleNamespace

from datetime import date
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
from scipy import ndimage

from ._model import UNet, Fcn
import laynet._metrics as lfs  # lfs=lossfunctions
from ._dataset import RsomLayerDataset, \
    RandomZShift, RandomZRescale, CropToEven, RandomMirror, IntensityTransform, \
    ToTensor, to_numpy, timing
from utils import save_nii
from ._metrics import MetricCalculator, smoothness_loss_new


class LayerNetBase:
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
                 decision_boundary=0.5,
                 model_type='unet',
                 dropout=True,
                 batch_size=1,
                 sliding_window_size=1,
                 DEBUG=False
                 ):

        self.logfile = None
        self.DEBUG = DEBUG
        if DEBUG:
            print('DEBUG MODE')
        self.model_depth = model_depth
        self.dirs = dirs
        self.out_pred_dir = dirs['out']
        self.decision_boundary = decision_boundary

        # DATASET
        self.pred_dataset = RsomLayerDataset(self.dirs['pred'],
                                             training=False,
                                             batch_size=batch_size,
                                             sliding_window_size=sliding_window_size,
                                             transform=transforms.Compose([
                                                 CropToEven(network_depth=self.model_depth),
                                                 ToTensor()])
                                             )

        self.pred_dataloader = DataLoader(self.pred_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=2,
                                          pin_memory=True)

        # ARGS
        self.args = SimpleNamespace()
        self.args.device = device
        self.args.dtype = torch.float32
        self.args.minibatch_size = batch_size
        self.args.non_blocking = True

        # MODEL
        if model_type['type'] == 'unet':
            self.model = UNet(in_channels=2,
                              n_classes=1,
                              depth=model_depth,
                              wf=model_type['wf'],
                              padding=True,
                              batch_norm=True,
                              up_mode='upconv',
                              dropout=dropout)
        elif model_type['type'] == 'fcn':

            self.model = Fcn()
        else:
            raise NotImplementedError

        self.model = self.model.to(self.args.device)
        self.model.float()

        if self.dirs['model']:
            self.model.load_state_dict(torch.load(self.dirs['model']))
            self.printandlog("Load model from", self.dirs['model'])
        else:
            self.printandlog("Did not load model.")

    def printandlog(self, *msg):
        print(*msg)
        if not self.DEBUG and self.logfile != None:
            with open(self.logfile, 'a') as fd:
                print(*msg, file=fd)

    def calc_metrics(self):
        results = self.metricCalculator.calculate(p=self.decision_boundary)
        self.printandlog("")
        self.printandlog("Metrics of eval set:")
        self.printandlog(json.dumps(results, indent=2))

    @timing
    def predict(self, model=None, save_all=True):
        self.metricCalculator = MetricCalculator()

        if model is None:
            model = self.model.eval()

        iterator = iter(self.pred_dataloader)
        futs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for i in range(len(self.pred_dataset)):
                # get the next volume to evaluate
                batch = next(iterator)

                prob_list = []
                for subsample in batch:
                    prediction = self._predict_one(batch=subsample, model=model)
                    # print(f'{prediction.shape=}')
                    prob_list.append((to_numpy(prediction, subsample['meta']), subsample['meta']))

                # save the single probabilities
                for el in prob_list:
                    data, meta = el
                #    if save_all:
                #        #futs.append(executor.submit(self._save_nii,
                #        #                            data,
                #        #                            meta=meta,
                #        #                            fstr='ppred.nii.gz'))
                #        futs.append(executor.submit(self._save_nii,
                #                                    data >= self.decision_boundary,
                #                                    meta=meta,
                #                                    fstr='pred.nii.gz'))

                # save combined one
                assert (len(prob_list) == 2)
                combined = (prob_list[0][0] + prob_list[1][0]) / 2

                #if save_all:
                    #futs.append(executor.submit(self._save_nii,
                    #                            combined,
                    #                            meta=meta,
                    #                            combined=True,
                    #                            fstr='ppred.nii.gz'))

                boolean_combined = combined >= self.decision_boundary
                #futs.append(executor.submit(self._save_nii,
                #                            boolean_combined,
                #                            meta=meta,
                #                            combined=True,
                #                            fstr='pred.nii.gz'))
                futs.append(executor.submit(self.postprocess_layerseg,
                                            boolean_combined,
                                            meta=meta,
                                            combined=True,
                                            fstr='pred.nii.gz'#'preds.nii.gz'
                                            ))
                # print(f"{batch[0]['label'].shape=}")

                # batch[0] and batch[1] have the same label
                self.metricCalculator.register_sample(
                    label=to_numpy(torch.squeeze(batch[0]['label'], dim=0), batch[0]['meta']),
                    prediction=combined,
                    name=os.path.basename(meta['filename'][0]))

            for _ in concurrent.futures.as_completed(futs):
                pass

    def postprocess_layerseg(self, vol, meta, combined, fstr):
        vol_shape = vol.shape
        structure = ndimage.generate_binary_structure(3, 2)

        pad_width = 6
        closing_iter1 = 5
        closing_iter2 = 1
        assert pad_width == closing_iter1 + closing_iter2

        vol = np.pad(vol, pad_width=pad_width, mode='edge')
        vol = ndimage.binary_closing(vol, structure=structure, iterations=closing_iter1, border_value=0)
        vol = ndimage.binary_opening(vol, structure=structure, iterations=5, border_value=0)
        vol = ndimage.binary_closing(vol, structure=structure, iterations=closing_iter2, border_value=0)
        vol = vol[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

        assert vol_shape == vol.shape

        self._save_nii(vol, meta, combined, fstr)

    @timing
    def _predict(self, model=None, save=True):
        self.metricCalculator = MetricCalculator()

        if model is None:
            model = self.model.eval()

        iterator = iter(self.pred_dataloader)

        for i in range(len(self.pred_dataset)):
            # get the next volume to evaluate
            batch = next(iterator)

            prob_list = []
            for subsample in batch:
                prediction = self._predict_one(batch=subsample, model=model)
                # print(f'{prediction.shape=}')
                prob_list.append((to_numpy(prediction, subsample['meta']), subsample['meta']))

            # save the single probabilities
            for el in prob_list:
                data, meta = el
                #if save:
                #    self._save_nii(data, meta=meta, fstr='ppred.nii.gz')
                #    self._save_nii(data >= self.decision_boundary, meta=meta, fstr='pred.nii.gz')

            # save combined one
            assert (len(prob_list) == 2)
            combined = (prob_list[0][0] + prob_list[1][0]) / 2

            if save:
                #self._save_nii(combined, meta=meta, combined=True, fstr='ppred.nii.gz')
                self._save_nii(combined >= self.decision_boundary, meta=meta, combined=True, fstr='pred.nii.gz')
            # print(f"{batch[0]['label'].shape=}")

            # batch[0] and batch[1] have the same label
            self.metricCalculator.register_sample(
                label=to_numpy(torch.squeeze(batch[0]['label'], dim=0), batch[0]['meta']),
                prediction=combined,
                name=os.path.basename(meta['filename'][0]))

    def _save_nii(self, data, meta, combined=False, fstr=''):
        filename = os.path.join(self.out_pred_dir, os.path.basename(meta['filename'][0]))
        if not combined:
            batch_axis = meta['batch_axis'][0]
            fstr = batch_axis + '_' + fstr
        else:
            fstr = fstr

        if 'ppred' in fstr:
            data = data.astype(np.float32)
            img_data = nib.Nifti1Image(data, np.eye(4))
            if not self.DEBUG:
                nib.save(img_data, filename.replace('rgb.nii.gz', fstr))
        else:
            data = data.astype(np.uint8)
            img_data = nib.Nifti1Image(data, np.eye(4))
            if not self.DEBUG:
                nib.save(img_data, filename.replace('rgb.nii.gz', fstr))

    def _predict_one(self, batch, model):
        batch['data'] = batch['data'].to(self.args.device,
                                         self.args.dtype,
                                         non_blocking=self.args.non_blocking)
        batch['data'] = torch.squeeze(batch['data'], dim=0)

        # divide into minibatches
        minibatches = np.arange(batch['data'].shape[0], step=self.args.minibatch_size)
        # init empty prediction stack

        shp = batch['data'].shape
        # [0 x 2 x 500 x 332]
        prediction_stack = torch.zeros((0, 1, shp[2], shp[3]),
                                       dtype=self.args.dtype,
                                       requires_grad=False
                                       )

        for i2, idx in enumerate(minibatches):
            data = batch['data'][idx:idx + self.args.minibatch_size, ...]

            prediction = model(data)

            prediction = prediction.detach()
            prediction = prediction.to('cpu')
            prediction_stack = torch.cat((prediction_stack, prediction), dim=0)

        # transform -> labels
        prediction_stack = torch.sigmoid(prediction_stack)

        return prediction_stack

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

            label = prediction_stack >= self.decision_boundary

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

            print('Saving', filename)
            if 1:
                save_nii(label.astype(np.uint8), self.out_pred_dir, filename + 'pred')

            if 1:
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
    """
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
    """

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
                 class_weight=None,
                 epochs=30,
                 dropout=False,
                 DEBUG=False,
                 decision_boundary=0.5,
                 batch_size=1,
                 aug_params=None,
                 loss_scheduler=None
                 ):

        super().__init__(dirs=dirs,
                         device=device,
                         model_depth=model_depth,
                         decision_boundary=decision_boundary,
                         model_type=model_type,
                         dropout=dropout,
                         batch_size=batch_size,
                         sliding_window_size=aug_params.sliding_window_size,
                         DEBUG=DEBUG)

        self.aug_params = aug_params
        self.sdesc = sdesc
        self.loss_scheduler = loss_scheduler

        self.LOG = True

        out_root_list = os.listdir(dirs['out'])

        today = date.today().strftime('%y%m%d')
        today_existing = [el for el in out_root_list if today in el]
        if today_existing:
            nr = max([int(el[7:9]) for el in today_existing]) + 1
        else:
            nr = 0

        self.today_id = today + '-{:02d}'.format(nr)
        self.dirs['out'] = os.path.join(self.dirs['out'], self.today_id)
        if self.sdesc:
            self.dirs['out'] += '-' + self.sdesc

        self.debug('Output directory string:', self.dirs['out'])

        # PROCESS LOGGING
        if not self.DEBUG:
            os.mkdir(self.dirs['out'])
            self.logfile = os.path.join(self.dirs['out'], 'log' + self.today_id)
        else:
            self.logfile = None

        if self.dirs['pred']:
            if not self.DEBUG:
                # overwrite out_pred_dir from superclass
                self.out_pred_dir = os.path.join(self.dirs['out'], 'prediction')
                os.mkdir(self.out_pred_dir)
            else:
                self.out_pred_dir = ''

        # MODEL
        self.model_dropout = dropout

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

        # DATASET
        self.train_dataset_zshift = dataset_zshift
        self._setup_dataloaders(batch_size=batch_size)

        # OPTIMIZER
        self.initial_lr = initial_lr
        self._setup_optimizer(optimizer=optimizer, scheduler_patience=scheduler_patience)

        # HISTORY
        self.history = {
            'train': {'epoch': [], 'loss': [], 'unred_loss': []},
            'eval': {'epoch': [], 'loss': [], 'unred_loss': []}
        }

        # ADDITIONAL ARGS
        self.args.n_epochs = epochs

        # TODO fix this
        self.args.data_dim = self.eval_dataset[0][0]['data'].shape

    def _setup_dataloaders(self, batch_size):

        self.train_dataset = RsomLayerDataset(self.dirs['train'],
                                              training=True,
                                              batch_size=batch_size,
                                              sliding_window_size=self.aug_params.sliding_window_size,
                                              transform=transforms.Compose([
                                                  RandomZRescale(p=0.3, range=(0.6, 1.4)),
                                                  RandomZShift(max_shift=self.aug_params.zshift),
                                                  RandomMirror(),
                                                  CropToEven(network_depth=self.model_depth),
                                                  IntensityTransform(),
                                                  ToTensor()])
                                              )

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=1,
                                           shuffle=True,
                                           drop_last=False,
                                           num_workers=6,
                                           pin_memory=True)

        self.eval_dataset = RsomLayerDataset(self.dirs['eval'],
                                             training=False,
                                             batch_size=batch_size,
                                             sliding_window_size=self.aug_params.sliding_window_size,
                                             transform=transforms.Compose([
                                                 CropToEven(network_depth=self.model_depth),
                                                 ToTensor()])
                                             )

        self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=2,
                                          pin_memory=True)

    def _setup_optimizer(self, *, optimizer, scheduler_patience):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.initial_lr,
                weight_decay=0)
        else:
            raise NotImplementedError

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

    def debug(self, *msg):
        if self.DEBUG:
            print(*msg)

    def _print(self, where):
        print('LayerUNET configuration:', file=where)
        print('DATA: train dataset loc:', self.dirs['train'], file=where)
        print('      train dataset len:', len(self.train_dataset), file=where)
        print('      eval dataset loc:', self.dirs['eval'], file=where)
        print('      eval dataset len:', len(self.eval_dataset), file=where)
        print('AUG:', file=where)
        print(self.aug_params, file=where)
        print('EPOCHS:', self.args.n_epochs, file=where)
        print('OPTIMIZER:', self.optimizer, file=where)
        print('initial lr:', self.initial_lr, file=where)
        print('LOSS: fn', self.lossfn, file=where)
        print('      class_weight', self.class_weight, file=where)
        print('CNN:  unet', file=where)
        print('      depth', self.model_depth, file=where)
        print('      dropout?', self.model_dropout, file=where)
        print('OUT:  model:', self.dirs['model'], file=where)
        print('      pred:', self.dirs['pred'], file=where)
        print('')
        print(self.model, file=where)

    def printConfiguration(self):
        self._print(sys.stdout)
        if not self.DEBUG and self.logfile != None:
            with open(self.logfile, 'a') as fd:
                self._print(fd)

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

            # torch.cuda.empty_cache()
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

            # print(torch.cuda.memory_cached()*1e-6,'MB memory used')
            # extract the average training loss of the epoch
            le_idx = self.history['train']['epoch'].index(curr_epoch)
            le_losses = self.history['train']['loss'][le_idx:]

            self.debug("N 2D slices train", (len(self.train_dataset) * self.train_dataset.batch_size))
            train_loss = sum(le_losses) / (len(self.train_dataset) * self.train_dataset.batch_size)
            # extract most recent eval loss
            curr_loss = self.history['eval']['loss'][-1]

            # use ReduceLROnPlateau scheduler
            self.scheduler.step(curr_loss)
            if self.loss_scheduler is not None:
                self.loss_scheduler.increase_epoch()

            if curr_loss < self.best_loss:
                self.best_loss = copy.deepcopy(curr_loss)
                self.best_model = copy.deepcopy(self.model.state_dict())
                for k, v in self.best_model.items():
                    self.best_model[k] = v.to('cpu')
                found_nb = 'new best!'
            else:
                found_nb = ''

            self.printandlog('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch + 1,
                self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb
            )

        print('finished training.')

    def train(self, iterator, epoch):
        self.model.train()
        for i in range(len(self.train_dataset)):
            # get the next batch of training data
            try:
                batch = next(iterator)
            except StopIteration:
                print('Iterators wrong')
                break

            label = batch['label'].to(
                self.args.device,
                dtype=self.args.dtype,
                non_blocking=self.args.non_blocking)
            data = batch['data'].to(
                self.args.device,
                self.args.dtype,
                non_blocking=self.args.non_blocking)

            label = torch.squeeze(label, dim=0)
            data = torch.squeeze(data, dim=0)

            # print(f"{data.shape=}")

            prediction = self.model(data)

            # print(f"{prediction.shape=}")

            # move back to save memory
            # prediction = prediction.to('cpu')
            if self.loss_scheduler is None:
                loss = self.lossfn(input=prediction, target=label)
                # self.debug(f"{loss=}")
                # sloss = smoothness_loss_new(prediction)
                # self.debug(f"{sloss=}")
            else:
                loss = self.loss_scheduler.loss(input=prediction, target=label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            frac_epoch = epoch + i / len(self.train_dataset)

            self.history['train']['epoch'].append(frac_epoch)
            self.history['train']['loss'].append(loss.data.item())
            del loss

    def eval(self, iterator, epoch):
        self.model.eval()
        running_loss = 0.0
        sizes = 0

        for i in range(len(self.eval_dataset)):
            # get the next batch of the evaluation set
            try:
                batch = next(iterator)
            except StopIteration:
                print('Iterators wrong')
                break

            for subsample in batch:
                running_loss += self._eval_one(subsample)
                sizes += subsample['data'].shape[0]

        epoch_loss = running_loss / sizes
        self.history['eval']['epoch'].append(epoch)
        self.history['eval']['loss'].append(epoch_loss)

        self.debug("N 2D slices eval", sizes)

    def _eval_one(self, batch):

        # print(f"{batch['data'].shape=}")

        batch['label'] = torch.squeeze(batch['label'], dim=0)
        batch['data'] = torch.squeeze(batch['data'], dim=0)

        # print(f"{batch['data'].shape=}")

        running_loss = 0.0

        # divide into minibatches
        # use same batch size as for training
        batch_size = self.train_dataset.batch_size
        minibatches = np.arange(batch['data'].shape[0], step=batch_size)
        for i, idx in enumerate(minibatches):
            data = batch['data'][idx:idx + self.args.minibatch_size, ...]
            label = batch['label'][idx:idx + self.args.minibatch_size, ...]

            label = label.to(self.args.device,
                             dtype=self.args.dtype,
                             non_blocking=self.args.non_blocking)
            data = data.to(self.args.device,
                           self.args.dtype,
                           non_blocking=self.args.non_blocking)

            # print(f"{data.shape=}")

            prediction = self.model(data)

            if self.loss_scheduler is None:
                loss = self.lossfn(input=prediction, target=label)
            else:
                loss = self.loss_scheduler.loss(input=prediction, target=label)

            # loss running variable
            # add value for every minibatch
            running_loss += loss.data.item()

        return running_loss

    def save_code_status(self):
        if not self.DEBUG:
            try:
                path = os.path.join(self.dirs['out'], 'git')
                os.system('git log -1 | head -n 1 > {:s}.diff'.format(path))
                os.system('echo /"\n/" >> {:s}.diff'.format(path))
                os.system('git diff >> {:s}.diff'.format(path))
            except:
                self.printandlog('Saving git diff FAILED!')

    def save_model(self, model='both', pat=''):
        if not self.DEBUG:
            if model == 'both':
                self.save_model(model='best', pat=pat)
                self.save_model(model='last', pat=pat)
            else:
                if model == 'best':
                    save_this = self.best_model
                elif model == 'last':
                    save_this = self.model.state_dict()
                else:
                    raise NotImplementedError

                torch.save(save_this, os.path.join(
                    self.dirs['out'],
                    'mod' + self.today_id + '_' + model + '_' + pat + '.pt'))

                json_f = json.dumps(self.history)
                f = open(os.path.join(self.dirs['out'], 'hist_' + self.today_id + pat + '.json'), 'w')
                f.write(json_f)
                f.close()