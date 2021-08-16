import torch
import warnings
# from torch import nn
import scipy.optimize
import copy
import numpy as np
import math
import matplotlib.pyplot as plt


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
    more = smoothness_weight * smoothness_loss(pred)

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
                                            padding=int(math.floor(window / 2)))

    label_conv = torch.squeeze(label_conv)
    label = torch.squeeze(label)

    # for perfectly smooth label, this value is zero
    # e.g. if label_conv[i] = label[i], -> 1/1 - 1 = 0
    label_smoothness = torch.abs(((label_conv + 1) / (label + 1)) - 1)

    # edge correction, steps at the boundaries do not count as unsmooth,
    # therefore corresponding entries of label_smoothness are zeroed out
    edge_corr = torch.zeros((pred_shape[3])).to('cuda')
    edge_corr[int(math.floor(window / 2)):-int(math.floor(window / 2))] = 1
    edge_corr = edge_corr.repeat(pred_shape[0] * pred_shape[2])

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
    label = (pred[:, 1, :, :] - pred[:, 0, :, :]).float()
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
                                            padding=int(math.floor(window / 2)))

    label_conv = torch.squeeze(label_conv)
    label = torch.squeeze(label)

    # for perfectly smooth label, this value is zero
    # e.g. if label_conv[i] = label[i], -> 1/1 - 1 = 0
    label_smoothness = torch.abs((label_conv + 1) / (label + 1) - 1)

    # edge correction, steps at the boundaries do not count as unsmooth,
    # therefore corresponding entries of label_smoothness are zeroed out
    edge_corr = torch.zeros((pred_shape[3])).to('cuda')
    edge_corr[int(math.floor(window / 2)):-int(math.floor(window / 2))] = 1
    edge_corr = edge_corr.repeat(pred_shape[0] * pred_shape[2])

    label_smoothness *= edge_corr

    # target shape
    # [minibatch x Z x X]

    # return some loss measure, as the sum of all smoothness losses
    return torch.sum(label_smoothness)


def calc_recall(*, label, pred):
    label = label.astype(np.bool)
    pred = pred.astype(np.bool)
    TP = np.sum(np.logical_and(label, pred))
    FN = np.sum(np.logical_and(label, np.logical_not(pred)))

    R = TP / (TP + FN)
    return R


def calc_precision(*, label, pred):
    label = label.astype(np.bool)
    pred = pred.astype(np.bool)
    TP = np.sum(np.logical_and(label, pred))
    FP = np.sum(np.logical_and(pred, np.logical_not(label)))

    P = TP / (TP + FP)
    return P


def _iou(*, label, pred):
    x = label
    y = pred
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = x.astype(np.bool)
    y = y.astype(np.bool)

    i = np.logical_and(x, y)
    return i.sum() / np.logical_or(x, y).sum()


def calc_dice(a, b):
    return _dice(a, b)


def _dice(*, label, pred):
    '''
    do the test in numpy
    '''
    x = label
    y = pred
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = x.astype(np.bool)
    y = y.astype(np.bool)

    i = np.logical_and(x, y)

    if x.sum() + y.sum() == 0:
        print('Dice No True values!')
        return 1.

    return (2. * i.sum()) / (x.sum() + y.sum())


class MetricCalculator:
    def __init__(self):
        self._labels = []
        self._preds = []
        self._names = []
        self._metrics_fn = dict()
        self._metrics_fn['dice'] = _dice
        self._metrics_fn['precision'] = calc_precision
        self._metrics_fn['recall'] = calc_recall
        self._metrics_fn['IoU'] = _iou

    def register_sample(self, *, label: np.ndarray, prediction: np.ndarray, name: str):
        self._labels.append(label)
        self._preds.append(prediction)
        self._names.append(name)

    # define function to optimize: 1 - metric
    def _optim_fun(self, p: float):
        results = self.calculate(p, metrics_fns={'dice': self._metrics_fn['dice']})
        return 1 - results['summary']['dice']['mean']

    def plot_dice(self, fname: str):
        # debug: produce plot showing x vs dice
        x_vec = np.linspace(0, 1, num=100)
        y_vec = np.vectorize(self._optim_fun)(x_vec)
        y_vec = 1 - y_vec  # dice score not dice loss

        fig, ax = plt.subplots()
        ax.plot(x_vec, y_vec)

        # ax.set_yscale('log')
        ax.set(xlabel='threshold', ylabel='dice')
        ax.grid()
        plt.savefig(fname)

    def optimize(self, metric: str = 'dice'):
        if metric != 'dice':
            raise NotImplementedError

        res = scipy.optimize.minimize_scalar(self._optim_fun, bounds=(0, 1), method='bounded')

        # return ideal p
        return res.x

    def calculate(self, p: float, metrics_fns=None):
        if metrics_fns is None:
            metrics_fns = self._metrics_fn

        results = dict()
        for label, pred, name in zip(self._labels, self._preds, self._names):
            for key, metric_fn in metrics_fns.items():
                score = metric_fn(label=label, pred=pred >= p)
                if key not in results:
                    results[key] = [(name, score)]
                else:
                    results[key].append((name, score))

        results = self._average(results)
        return results

    @staticmethod
    def _average(results):
        summary = dict()
        for metric_name, values in results.items():
            metric_list = []
            for _, value in values:
                metric_list.append(value)

            summary[metric_name] = dict()
            summary[metric_name]['mean'] = np.mean(metric_list)
            summary[metric_name]['std'] = np.std(metric_list)

        results['summary'] = summary
        return results


if __name__ == '__main__':

    mC = MetricCalculator()
    for _ in range(5):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        mC.register_sample(label=a, prediction=b, name="bla0")
    results = mC.calculate(p=0.5)

    p_ideal = mC.optimize()

