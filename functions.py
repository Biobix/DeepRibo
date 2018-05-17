import collections
from fitmodule.utils import ProgressBar
import math
import numpy as np
import torch
import json
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.nn import Module
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data.dataloader import numpy_type_map
from sklearn.metrics import average_precision_score, roc_auc_score


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
                        # Maintains max of all exp. moving avg. of sq. grad. values
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
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss


def auc_from_tensors(y_hat, y_true):
    y_true, y_hat = y_true.numpy(), y_hat.numpy()
    auc = roc_auc_score(y_true, y_hat[:, 1])
    return auc


def default_collate(batch):
    'Puts each data field into a tensor with outer dimension batch size'

    error_msg = 'batch must contain tensors, numbers, dicts or lists; found {}'
    _use_shared_memory = True
    string_classes = (str, bytes)
    int_classes = int

    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        pad = False
        out = None

        if not np.all([batch[0].shape == tensor.shape for tensor in batch]):
            pad = True
            batch_lens = np.array([len(tensor) for tensor in batch])
            max_len = np.max(batch_lens)
            out_batch = torch.zeros(len(batch), int(max_len),
                                    len(batch[0].shape))

            for i, variable in enumerate(batch):
                length = variable.size(0)
                out_batch[i, :length, :] = variable
            batch = out_batch

        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)

        if pad:
            return torch.stack(batch, dim=0, out=out), torch.from_numpy(batch_lens)
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
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]


def extend_lib(df, pred):
    df = df.copy()
    df['label'] = df['label'].values.astype(bool)
    df['in_gene'] = df['in_gene'].values.astype(bool)
    df['pred'] = pred[:, 1]
    df['gene_names'] = df.gene_names.str.lower()
    sort_idx = np.argsort(df['pred'].values)
    df.loc[sort_idx, 'pred_disc'] = np.arange(len(df))[::-1]
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
    df['dist'] = dist
    SS_pred_disc = np.full(len(df), 999999, dtype=np.int)
    sort_idx = df[df['SS']].sort_values(by='pred').index.values[::-1]
    SS_pred_disc[sort_idx] = np.arange(len(df[df['SS']]))
    df['SS_pred_disc'] = SS_pred_disc

    return df


class FitModule(Module):

    def fit(self,
            train_loader,
            test_loader=None,
            valid_loaders=None,
            valid_keys=None,
            scheduler=None,
            epochs=1,
            initial_epoch=0,
            seed=None,
            loss=None,
            optimizer=None,
            custom_metric=None,
            log=None,
            dest='default',
            verbose=1,
            RNN=False,
            HYBRID=False,
            GPU=False):
        '''Trains the model similar to Keras' .fit(...) method

        # Arguments
            X: training data Tensor.
            y: target data Tensor.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
                over the training data arrays.
            verbose: 0, 1. Verbosity mode.
                0 = silent, 1 = verbose.
            test_split: float between 0 and 1:
                fraction of the training data to be used as test data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
            test_data: (x_test, y_test) tuple on which to evaluate
                the loss and any model metrics
                at the end of each epoch. The model will not
                be trained on this data.
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            seed: random seed.
            optimizer: training optimizer
            loss: training loss
            metrics: list of functions with signatures `metric(y_true, y_pred)`
                where y_true and y_pred are both Tensors

        # Returns
            list of OrderedDicts with training metrics
        '''
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
        self.train()
        for t in range(initial_epoch, epochs):
            if scheduler:
                scheduler.step()
            print('Epoch {0} / {1}'.format(t+1, epochs))
            # Setup logger
            if verbose:
                pb = ProgressBar(len(train_loader))
            epoch_loss = 0.0
            # Run batches
            for batch_i, b_data in enumerate(train_loader):
                # Backprop
                opt.zero_grad()
                if HYBRID:
                    X_batch_RNN_len, sort_order = torch.sort(b_data[1][1],
                                                             descending=True)
                    y_batch = Variable(b_data[2][sort_order].type(dtypeY))
                    X_batch_conv = Variable(b_data[0][sort_order]
                                            .type(dtypeX).transpose_(0, 1))
                    X_batch_RNN = Variable(b_data[1][0][sort_order]
                                           .type(dtypeX).transpose_(0, 1))
                    X_batch_RNN_len = list(b_data[1][1][sort_order])
                    X_batch_RNN = pack_padded_sequence(X_batch_RNN,
                                                       X_batch_RNN_len)
                    y_batch_pred, hidden = self((X_batch_conv, X_batch_RNN))

                elif RNN:
                    y_batch = Variable(b_data[1].type(dtypeY))
                    X_batch = Variable(b_data[0].type(dtypeX).transpose_(0, 1))
                    hidden = self.init_hidden(len(X_batch[0]), dtypeX)
                    y_batch_pred, hidden = self(X_batch, hidden)
                else:
                    y_batch = Variable(b_data[1].type(dtypeY))
                    X_batch = Variable(b_data[0].type(dtypeX))
                    y_batch_pred = self(X_batch)
                batch_loss = loss(y_batch_pred, y_batch)
                batch_loss.backward()
                opt.step()
                # Update status
                epoch_loss += batch_loss.data[0]
                log.log_loss(batch_loss.data.cpu().numpy()[0])

                if verbose:
                    pb.bar(batch_i, log.output_metric())
            # Run metrics
            y_pred, y_true = self.predict(train_loader, RNN=RNN, HYBRID=HYBRID,
                                          log=log, GPU=GPU)
            log.log_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy())
            if test_loader is not None:
                y_pred, y_true = self.predict(test_loader, loss=loss, RNN=RNN,
                                              key='test', HYBRID=HYBRID,
                                              log=log, GPU=GPU)
                log.log_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy(),
                                'test')
            if valid_loaders is not None:
                for valid_loader, valid_key in zip(valid_loaders, valid_keys):
                    y_pred, y_true = self.predict(valid_loader, loss=loss,
                                                  RNN=RNN, key=valid_key,
                                                  HYBRID=HYBRID, log=log,
                                                  GPU=GPU)
                    log.log_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy(),
                                    valid_key)
            if verbose:
                pb.close()
            torch.save(self, '{}_{}'.format(dest, t))
            with open('{}_{}.json'.format(dest, t), 'w') as fp:
                json.dump(log.metrics, fp)
            log.output_metrics()
        return log

    def predict(self, loader, loss=None, RNN=False, key=None, HYBRID=False,
                log=None, GPU=False):
        '''Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            X: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
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
        if loader.sampler:
            n = len(loader.sampler)
        batch_size = loader.batch_size
        for b_data in loader:
            # Predict on batch
            if HYBRID:
                X_batch_RNN_len, sort_order = torch.sort(b_data[1][1],
                                                         descending=True)
                revert_mask = np.argsort(sort_order.numpy())
                y_batch = Variable(b_data[2].type(dtypeY), volatile=True)
                X_batch_conv = Variable(b_data[0][sort_order].type(dtypeX)
                                        .transpose_(0, 1), volatile=True)
                X_batch_RNN = Variable(b_data[1][0][sort_order].type(dtypeX)
                                       .transpose_(0, 1), volatile=True)
                X_batch_RNN_len = list(b_data[1][1][sort_order])
                X_batch_RNN = pack_padded_sequence(X_batch_RNN,
                                                   X_batch_RNN_len)
                y_batch_pred, hidden = self((X_batch_conv, X_batch_RNN))
                if GPU:
                    y_batch_pred = y_batch_pred[torch.cuda.LongTensor(revert_mask)]
                else:
                    y_batch_pred = y_batch_pred[torch.LongTensor(revert_mask)]

            elif RNN:
                X_batch = Variable(b_data[0].type(dtypeX).transpose_(0, 1),
                                   volatile=True)
                hidden = self.init_hidden(len(X_batch[0]), dtypeX)
                y_batch_pred, hidden = self(X_batch, hidden)
            else:
                X_batch = Variable(b_data[0].type(dtypeX), volatile=True)
                y_batch_pred = self(X_batch)
            if key:
                batch_loss = loss(y_batch_pred, y_batch)
                log.log_loss(batch_loss.data.cpu().numpy()[0], key)
            # Infer prediction shape
            y_batch_pred = y_batch_pred.data
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
                y_true = torch.zeros((n,) + y_batch.data.size()[1:])
            # Add to prediction tensor
            y_pred[r: min(n, r + batch_size)] = y_batch_pred
            y_true[r: min(n, r + batch_size)] = y_batch.data
            r += batch_size
        return y_pred, y_true


class logger(object):
    def __init__(self, metrics, test=True, valid_keys=None):
        self.i = {'train': 0}
        self.log_auc, self.log_acc, self.log_p_r = False, False, False
        self.metrics = {'train': {}}
        if test:
            self.metrics['test'] = {}
            self.i['test'] = 0
        if valid_keys is not None:
            for valid_key in valid_keys:
                self.metrics[valid_key] = {}
                self.i[valid_key] = 0
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

        for key in self.metrics:
            self.metrics[key].update({'loss': [0]})

    def log_loss(self, loss, key='train'):
        self.i[key] += 1

        update = (self.metrics[key]['loss'][-1]*(self.i[key]-1)
                  + loss)/self.i[key]
        self.metrics[key]['loss'].append(update)

    def log_metrics(self, y_true, y_hat, key='train'):

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
        return self.metrics[key][metric][-1]

    def output_metrics(self):
        print('')
        for key in sorted(self.metrics):
            print('{}:'.format(key), end='')
            for k, v in self.metrics[key].items():
                print('\t{}: {:5.3f}'.format(k, v[-1]), end='')
            print('\n', end='')
