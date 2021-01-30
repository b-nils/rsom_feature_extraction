import os
import numpy as np
import copy
import warnings
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from ._model import DeepVesselNet
from ._dataset import RSOMVesselDataset, \
        DropBlue, ToTensor, to_numpy

from .patch_handling import get_volume

class VesNetBase():
    """
    stripped base class for predicting RSOM vessels.
    for training use class VesNet
    """
    def __init__(self,
                 dirs={'train':'','eval':'', 'model':'', 'pred':''}, #add out
                 device=torch.device('cuda'),
                 model=None,
                 divs = (4, 4, 3),
                 offset = (6, 6, 6),
                 batch_size = 1,
                 ves_probability=0.5,
                 ):

        self.DEBUG = False

        # OUTPUT DIRECTORIES
        self.dirs = dirs

        # MODEL
        if model is not None:
            self.model = model
        else:
            self.model = DeepVesselNet(groupnorm=True)
        
        if self.dirs['model']:
            print('Loading VesNet model from:', self.dirs['model'])
            try:
                self.model.load_state_dict(torch.load(self.dirs['model']))
            except:
                warnings.warn('Could not load model!', UserWarning) 

        self.out_pred_dir = self.dirs['out']

        self.model = self.model.to(device)
        self.model = self.model.float()

        # VESSEL prediction probability boundary
        self.ves_probability = ves_probability

        # DIVS, OFFSET
        self.divs = divs
        self.offset = offset

        # DATASET
        self._setup_dataloaders()
        self.batch_size = batch_size

        # ADDITIONAL ARGS
        self.non_blocking = True
        self.device = device
        self.dtype = torch.float32

    def _setup_dataloaders(self):

        if self.dirs['pred']:
            self.pred_dataset = RSOMVesselDataset(self.dirs['pred'],
                                              divs=self.divs,
                                              offset=self.offset,
                                              transform=transforms.Compose([
                                                  DropBlue(),
                                                  ToTensor()]))

            self.pred_dataloader = DataLoader(self.pred_dataset,
                                              batch_size=1, # easier for reconstruction 
                                              shuffle=False, 
                                              num_workers=4,
                                              pin_memory=True)
        
        self.data_shape = self.pred_dataset[0]['data'].shape
        if self.dirs['pred']:
            self.size_pred = len(self.pred_dataset)

    def predict(self, 
                use_best=True, 
                cleanup=True,
                save_ppred=True):
        '''
        doc string missing
        '''
        # TODO: better solution needed?
        if use_best:
            # print('Using best model.')
            self.model.load_state_dict(self.best_model)
        else:
            # print('Using last model.')
            pass

        iterator = iter(self.pred_dataloader) 
        self.model.eval()

        prediction_stack = []
        index_stack = []

        for i in range(self.size_pred):
            # get the next batch of the evaluation set
            batch = next(iterator)
            
            data = batch['data'].to(
                    self.device,
                    self.dtype,
                    non_blocking=self.non_blocking)
            
            debug('prediction, data shape:', data.shape)
            torch.cuda.empty_cache()                
            prediction = self.model(data)
            
            debug(torch.cuda.max_memory_allocated()/1e6, 'MB memory used') 
            prediction = prediction.detach()
            # convert to probabilities
            sigmoid = torch.nn.Sigmoid()
            prediction = sigmoid(prediction)

            # otherwise can't reconstruct.
            if i==0:
                assert batch['meta']['index'].item() == 0
             
            prediction_stack.append(prediction)
            index_stack.append(batch['meta']['index'].item())
            
            # if we got all patches
            if batch['meta']['index'] == np.prod(self.divs) - 1:
                
                debug('Reconstructing volume: index stack is:')
                debug(index_stack)

                assert len(prediction_stack) == np.prod(self.divs)
                assert index_stack == list(range(np.prod(self.divs)))
                
                patches = (torch.stack(prediction_stack)).to('cpu').numpy()
                prediction_stack = []
                index_stack = []

                debug('patches shape:', patches.shape)
                patches = patches.squeeze(axis=(1,2))
                debug('patches shape:', patches.shape)
                
                V = get_volume(patches, self.divs, (0,0,0))
                V = to_numpy(V, batch['meta'], Vtype='label', dimorder='torch')
                debug('reconstructed volume shape:', V.shape)

                debug('vessel probability min/max:', np.amin(V),'/', np.amax(V))

                # binary cutoff
                Vbool = V >= self.ves_probability

                # save to file
                if not self.DEBUG:
                    if os.path.exists(self.dirs['out']):
                        # create ../prediction directory
                        dest_dir = os.path.join(self.out_pred_dir)
                        fstr = batch['meta']['filename'][0].replace('.nii.gz','')  + '_pred'
                        self.saveNII(Vbool.astype(np.uint8), dest_dir, fstr)
                        #print('Saving vessel prediction', fstr)
                        if save_ppred:
                            fstr = fstr.replace('_pred', '_ppred')
                            self.saveNII(V, dest_dir, fstr)
                    else:
                        print('Couldn\'t save prediction.')
    @staticmethod
    def saveNII(V, path, fstr):
        img = nib.Nifti1Image(V, np.eye(4))
    
        fstr = fstr + '.nii.gz'
        nib.save(img, os.path.join(path, fstr))

    def printandlog(self, *msg):
        print(*msg)

def debug(*msg):
    ''' debug print helper function'''
    if 'DEBUG' in globals():
        if DEBUG:
            print(*msg)




