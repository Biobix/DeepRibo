#####################################
# DeepRibo: precise gene annotation of prokaryotes using deep learning
# and ribosome profiling data
#
#  Copyright (C) 2018 J. Clauwaert, G. Menschaert, W. Waegeman
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  For more (contact) information visit http://www.biobix.be/DeepRibo
#####################################

import sys
import collections
import re
import argparse
import math
import torch
import json
import datetime as dt
import numpy as np
from torch.optim import Optimizer
from torch.nn import Module
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import numpy_type_map
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data.sampler import Sampler


class Adam(Optimizer):
    '''Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        '''Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        '''
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients'
                                       ', please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad.
                        # values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till
                    # now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (math.sqrt(bias_correction2) /
                                           bias_correction1)

                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss


class BatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class BucketSampler(Sampler):
    """Samples elements in buckets, bucketShuffle needs to be called every epoch.

    Arguments:
        data_source (Dataset): dataset to sample from
        index (list/numpy): array with index locations
        batch_size (int): batch size
    """

    def __init__(self, data_source, index, batch_size):
        starts = data_source.loc[index, 'start_site']
        stops = data_source.loc[index, 'stop_site']
        self.lens = abs(starts-stops)
        self.batch_size = batch_size
        self.s_idx = np.argsort(self.lens)
        self.sort_idx = np.array(self.lens.index[self.s_idx])
        self.bucketShuffle()

    def __iter__(self):
        return iter(self.idx_list)

    def __len__(self):
        return len(self.lens)

    def bucketShuffle(self):
        # shuffle rows within regions
        region_size = np.maximum(int(len(self.lens)/self.batch_size//12), 100)
        inc_batch_reg = len(self.lens) % region_size
        index_list = np.array(self.sort_idx[inc_batch_reg:])
        np.random.shuffle(np.reshape(index_list, (-1, region_size)).T)
        shuffled_index = np.hstack((index_list, self.sort_idx[:inc_batch_reg]))

        # shuffle batches within dataset
        inc_batch = len(self.lens) % self.batch_size
        shuffled_index_list = shuffled_index[inc_batch:]
        np.random.shuffle(np.reshape(shuffled_index_list,
                                     (-1, self.batch_size)))

        # set dataset with new order
        self.idx_list = np.hstack((shuffled_index_list,
                                   shuffled_index[:inc_batch]))


def aucFromTensors(y_hat, y_true):
    y_true, y_hat = y_true.numpy(), y_hat.numpy()
    auc = roc_auc_score(y_true, y_hat[:, 1])
    return auc


def defaultCollate(batch):
    '''Puts each data field into a tensor with outer dimension batch size'
    code copied from
    https://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html#DataLoader
    and tweaked for personal use'''

    error_msg = 'batch must contain tensors, numbers, dicts or lists; found {}'
    _use_shared_memory = True
    string_classes = (str, bytes)
    int_classes = int

    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        pad = False
        out = None

        # if not np.all([batch[0].shape == tensor.shape for tensor in batch]):
        if batch[0].shape[0] != 4:
            pad = True
            batch_lens = np.sort([b.shape[0] for b in batch])[::-1].copy()
            sort_order = np.argsort([b.shape[0] for b in batch])[::-1].copy()
            batch = pad_sequence([batch[idx] for idx in sort_order])

            batch.unsqueeze_(2).contiguous()

        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)

        if pad:
            # return torch.stack(batch, dim=0, out=out),
            # torch.from_numpy(batch_lens)
            return (batch, batch_lens, sort_order)
        else:
            return torch.stack(batch, dim=0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: defaultCollate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [defaultCollate(samples) for samples in transposed]


def extendLib(df, pred):
    '''Function that uses the predictions to extend the data_list.csv object
    of a given dataset

    Attributes:
        df (DataFrame): dataframe containing data_list.csv table created by
            DataParser.py
        pred (numpy array): output array produced by the neural network
    '''
    df = df.copy()
    df['label'] = df['label'].values.astype(bool)
    df['in_gene'] = df['in_gene'].values.astype(bool)
    df['pred'] = pred[:, 1]
    sort_idx = np.argsort(df['pred'].values)
    df.loc[sort_idx, 'pred_rank'] = np.arange(len(df))[::-1]
    SS = np.full(len(df), False)
    dist = np.zeros(len(df))
    for strand in df['strand'].unique():
        for stop in df[df['strand'] == strand]['stop_site'].unique():
            mask = np.where((df['strand'] == strand) &
                            (df['stop_site'] == stop))[0]
            if len(mask) > 1:
                SS[df.loc[mask, 'pred'].idxmax()] = True
                dist_mask = df.loc[mask, 'label'] == False
                if np.any(dist_mask == False):
                    right = df.loc[mask].loc[dist_mask == False,
                                             'start_site'].iloc[0]
                    left = df.loc[mask].loc[dist_mask, 'start_site'].values

                    dist[mask[dist_mask]] = left - right
                else:
                    dist[mask[dist_mask]] = -1
            else:
                SS[mask] = True
                if not df.loc[mask, 'label'].values:
                    dist[mask] = -1

    df['SS'] = SS
    df['dist'] = dist.astype(np.int)
    SS_pred_rank = np.full(len(df), 999999, dtype=np.int)
    sort_idx = df[df['SS']].sort_values(by='pred').index.values[::-1]
    SS_pred_rank[sort_idx] = np.arange(len(df[df['SS']]))
    df['SS_pred_rank'] = SS_pred_rank

    return df


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class FitModule(Module):
    def fit(self, device, train_loader, valid_loader=None, test_loaders=None,
            test_keys=None, scheduler=None, epochs=50, initial_epoch=0,
            seed=None, loss=None, optimizer=None, log=None, dest='default',
            verbose=1, GPU=False):
        '''Trains the model similar to Keras' .fit(...) method

        Arguments:
            train_loader (DataLoader): data loader for training
            valid_loader (DataLoader): data loader for validation
            test_loaders (list): list containing DataLoader objects
                which are all used for testing at the end of an
                epoch of training
            test_keys (list): list containing labels for each of
                the valid_loaders
            scheduler (object): object used for gradational decrease
                of the learning stop during training
            epochs (int): training epochs (default: 50)
            initial_epoch (int): epoch at which to start training
                (useful for resuming a previous training run)
                over the training data arrays.
            seed (int): random seed.
            loss (object): training loss
            optimizer (object): training optimizer
            log (Logger object): Logger object with which training/testing
                metrics are processed/saved
            dest (string): path to which trained model weights are saved
            verbose (0,1): verbosity mode; 0 = silent, 1 = verbose
            GPU (bool): trains using a GPU


        Returns:
            Logger object with training metrics
        '''
        ts = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if seed and seed >= 0:
            torch.manual_seed(seed)
        # Prepare test data
        if GPU:
            dtypeX = torch.cuda.FloatTensor
            dtypeY = torch.cuda.LongTensor
        else:
            dtypeX = torch.FloatTensor
            dtypeY = torch.LongTensor
        # Compile optimizer
        opt = optimizer
        # Run training loop
        for t in range(initial_epoch, epochs):
            if scheduler:
                scheduler.step()
            print('Epoch {0} / {1}'.format(t+1, epochs))
            # Setup Logger
            pb = ProgressBar(len(train_loader), verbose=verbose)
            epoch_loss = 0.0
            # Run batches
            self.train()
            for b_i, b_data in enumerate(train_loader):
                # Backprop
                opt.zero_grad()

                sort_order, X_batch_RNN_len = b_data[1][2], b_data[1][1]
                y_batch = b_data[2][sort_order].type(dtypeY).to(device)
                X_batch_conv = b_data[0][sort_order].type(dtypeX).to(device)
                X_batch_RNN = b_data[1][0].type(dtypeX).to(device)
                X_batch_RNN = pack_padded_sequence(X_batch_RNN,
                                                   X_batch_RNN_len)
                y_batch_pred, hidden = self((X_batch_conv, X_batch_RNN))

                batch_loss = loss(y_batch_pred, y_batch)
                batch_loss.backward()
                opt.step()
                # Update status
                epoch_loss += batch_loss.item()
                log.log_loss(batch_loss.item())

                pb.bar(b_i, log.output_metric())
            pb.close()
            # Reshuffle bucket sampler
            train_loader.batch_sampler.sampler.bucketShuffle()
            # Run metrics
            y_pred, y_true = self.predict(device, train_loader, log=log,
                                          GPU=GPU, verbose=verbose)
            log.log_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy())
            if valid_loader is not None:
                y_pred, y_true = self.predict(device, valid_loader, loss=loss,
                                              key='valid', log=log, GPU=GPU,
                                              verbose=verbose)
                log.log_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy(),
                                'valid')
            if test_loaders is not None:
                for test_loader, test_key in zip(test_loaders, test_keys):
                    y_pred, y_true = self.predict(device, test_loader,
                                                  loss=loss, key=test_key,
                                                  log=log, GPU=GPU,
                                                  verbose=verbose)
                    log.log_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy(),
                                    test_key)
            torch.save(self.state_dict(), '{}_{}_epoch_{}.pt'.format(dest, ts,
                                                                     t+1))
            with open('{}_{}_{}.json'.format(dest, ts, t), 'w') as fp:
                json.dump(log.metrics, fp)
            log.output_metrics()
        return log

    def predict(self, device, loader, loss=None, key=None, log=None, GPU=False,
                verbose=False):
        '''Generates output predictions for the input samples.
        Computation is done in batches.

        Arguments:
            loader (DataLoader): loader for data
            loss (object): training loss object
            key (string): label of processed data
            log (Logger object): logger object with which training/testing
                metrics are processed/saved
            GPU (bool): trains using a GPU
        '''
        # Build DataLoader
        # Batch prediction
        if GPU:
            dtypeX = torch.cuda.FloatTensor
            dtypeY = torch.cuda.LongTensor
        else:
            dtypeX = torch.FloatTensor
            dtypeY = torch.LongTensor

        self.eval()
        r = 0
        n = len(loader.batch_sampler.sampler)
        batch_size = loader.batch_sampler.batch_size
        pb = ProgressBar(len(loader), verbose=verbose)
        for b_i, b_data in enumerate(loader):
            # Predict on batch
            with torch.no_grad():
                sort_order, X_batch_RNN_len = b_data[1][2], b_data[1][1]
                y_batch = b_data[2][sort_order].type(dtypeY).to(device)
                X_batch_conv = b_data[0][sort_order].type(dtypeX).to(device)
                X_batch_RNN = b_data[1][0].type(dtypeX).to(device)
                X_batch_RNN = pack_padded_sequence(X_batch_RNN,
                                                   X_batch_RNN_len)
                y_batch_pred, hidden = self((X_batch_conv, X_batch_RNN))

            if key:
                batch_loss = loss(y_batch_pred, y_batch)
                log.log_loss(batch_loss.item(), key)
            # Infer prediction shapei
            y_batch_pred = y_batch_pred.data
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
                y_true = torch.zeros((n,) + y_batch.data.size()[1:])
            # Add to prediction tensor
            unsort_idx = np.argsort(sort_order)
            y_pred[r: min(n, r + batch_size)] = y_batch_pred[unsort_idx]
            y_true[r: min(n, r + batch_size)] = y_batch.data[unsort_idx]
            r += batch_size
            pb.bar(b_i)
        unbucket_idx = np.argsort(loader.batch_sampler.sampler.idx_list)
        pb.close()

        return y_pred[unbucket_idx], y_true[unbucket_idx]


class Logger(object):
    def __init__(self, args, metrics, valid=True, test_keys=None):
        '''Object which stores and calculates metrics produced by a neural
        network during training

        Attributes:
            metrics (list): lists all metrics stored during training. list can
                include ['acc','AUC', 'loss', 'P-R']
            valid (bool): store metrics of test set (default: True)
            test_keys (list): list of labels for each valid_loader used during
            training
        '''
        self.i = {'train': 0}
        self.log_auc, self.log_acc, self.log_p_r = False, False, False
        self.metrics = {'train': {}}
        if valid:
            self.metrics['valid'] = {}
            self.i['valid'] = 0
        if test_keys is not None:
            for test_key in test_keys:
                self.metrics[test_key] = {}
                self.i[test_key] = 0
        if 'acc' in metrics:
            self.log_acc = True
            for key in self.metrics:
                self.metrics[key].update({'acc': []})
        if 'AUC' in metrics:
            self.log_auc = True
            for key in self.metrics:
                self.metrics[key].update({'auc': []})
        if 'P-R' in metrics:
            self.log_p_r = True
            for key in self.metrics:
                self.metrics[key].update({'p-r': []})
        self.keys = list(self.metrics.keys())
        for key in self.keys:
            self.metrics[key].update({'loss': [0]})
        self.metrics['args'] = args

    def log_loss(self, loss, key='train'):
        '''Logs loss metric

        Attributes:
            loss (float): training loss
            key (string): label of processed data
        '''
        self.i[key] += 1

        update = (self.metrics[key]['loss'][-1]*(self.i[key]-1)
                  + loss)/self.i[key]
        self.metrics[key]['loss'].append(update)

    def log_metrics(self, y_true, y_hat, key='train'):
        '''Logs non-loss metrics

        Attributes:
            y_true (array): array containing true labels
            y_hat (array): array containing predicted labels
            key (string): label of processed data
        '''
        if self.log_auc:
            auc = roc_auc_score(y_true, y_hat[:, 1])
            self.metrics[key]['auc'].append(auc)
        if self.log_acc:
            acc = sum(np.argmax(y_hat, axis=1) == y_true)/len(y_true)
            self.metrics[key]['acc'].append(acc)
        if self.log_p_r:
            p_r = average_precision_score(y_true, y_hat[:, 1])
            self.metrics[key]['p-r'].append(p_r)

    def output_metric(self, key='train', metric='loss'):
        '''Prints last recorded value of metric

        Attributes:
            key (string): label of processed data
            metric (string): key of metric to be printed
        '''
        return self.metrics[key][metric][-1]

    def output_metrics(self):
        '''Prints last recorded values of all metrics
        '''
        print('')
        for key in sorted(self.keys):
            print('{}:'.format(key), end='')
            for k, v in self.metrics[key].items():
                print('\t{}: {:5.3f}'.format(k, v[-1]), end='')
            print('\n', end='')


class ProgressBar(object):
    """Cheers @ajratner"""

    def __init__(self, n, length=40, verbose=True):
        # Protect against division by zero
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length
        self.verbose = verbose
        # Precalculate the i values that should trigger a write operation
        self.ticks = [round(i/100.0 * n) for i in range(101)]
        self.ticks.append(n-1)
        self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, n-1]"""
        if i in self.ticks:
            if self.verbose:
                b = int(np.ceil(((i+1) / self.nf) * self.length))
                sys.stdout.write("\r[{0}{1}] {2}%\t{3}".format(
                    "="*b, " "*(self.length-b), int(100*((i+1) / self.nf)),
                    message))
            else:
                sys.stdout.write("=")
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n-1)
        sys.stdout.write("{0}\n".format(message))
        sys.stdout.flush()
