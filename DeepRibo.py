import math
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from functions import FitModule, default_collate, logger, Adam, extend_lib
s_curve_dict = {"bac": (0.003447865*82.465153, 0.10),
                "sau": (0.043717797*8.576148, 0.12),
                "sal": (0.004607822*57.277894, 0.11),
                "cre": (0.045049202*5.537422, 0.11),
                "coe": (0.077304740*3.527902, 0.12),
                "eco": (0.001911246*69.846616, 0.07)}


class CustomLoader(Dataset):
    def __init__(self, data_path, data, cutoff):
        self.data_path = data_path
        mask = np.array([])
        amask = np.array([])
        data_list = ["{}/{}/data_list.csv".format(data_path, d) for d in data]
        print(data_list)
        tmp_df = pd.concat([pd.read_csv(dl) for dl in data_list])
        for x in zip(data_list, cutoff[0], cutoff[1]):
            tmp_s_df = pd.read_csv(x[0])
            cond_1 = tmp_s_df["rpk_elo"] >= x[1]
            cond_2 = tmp_s_df["coverage_elo"] >= x[2]
            mask = np.hstack((mask, np.logical_and(cond_1, cond_2)))
            amask = np.hstack((amask, np.logical_or(cond_1 == False,
                                                    cond_2 == False)))
        self.cond_1 = mask
        self.anti_cond_1 = amask

        self.cond_2 = abs(tmp_df["start_site"]-tmp_df["stop_site"]) > 30
        self.mask = np.logical_and(self.cond_1, self.cond_2)
        self.anti_mask = np.logical_and(self.anti_cond_1, self.cond_2)

        self.list = tmp_df
        self.masked_list = tmp_df[self.mask].reset_index(drop=True)
        list_neg = tmp_df[self.anti_mask][tmp_df[self.anti_mask]["label"] == 1]

        con_1 = tmp_df["strand"].isin(list_neg["strand"])
        con_2 = tmp_df["stop_site"].isin(list_neg["stop_site"])
        temp_mask = np.logical_and(con_1, con_2) == False
        self.mask = np.logical_and(self.mask, temp_mask)
        self.masked_list = tmp_df[self.mask].reset_index(drop=True)

        X_train = tmp_df.loc[:, 'filename':'filename_counts'][self.mask]
        self.X_train = X_train.reset_index(drop=True)
        self.y_train = tmp_df['label'][self.mask].values

    def __getitem__(self, index):
        path = '{}/{}'.format(self.data_path, self.X_train.iloc[index, 0])
        img = torch.from_numpy(np.load(path)).contiguous()
        img = img.view(1, img.shape[0],
                       img.shape[1]).transpose(0, 2)
        path = "{}/{}".format(self.data_path, self.X_train.iloc[index, 1])
        counts = torch.from_numpy(np.load(path))
        label = self.y_train[index]
        return img, counts, label

    def __len__(self):
        return len(self.X_train.index)


class DualComplex(FitModule):
    def __init__(self, batch_size, hidden_size, layers,
                 bidirect=False):
        super(DualComplex, self).__init__()
        self.gru = nn.GRU(1, hidden_size=hidden_size,
                          num_layers=layers, bidirectional=bidirect)
        self.batch_size = batch_size
        self.layers = layers
        self.hidden_size = hidden_size
        self.input_len = 30
        self.bi = 2**bidirect
        self.conv_ch_1 = nn.Conv2d(4, 4, (1, 1))
        self.conv1 = nn.Conv2d(4, 32, (12, 1))
        self.fc1 = nn.Linear(self.hidden_size*layers*self.bi +
                             math.ceil((((self.input_len-11)))*32), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x, hidden=None):

        y = x[1]
        y, hidden = self.gru(y)
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(-1, self.hidden_size*self.layers*self.bi)
        x = x[0].transpose(0, 1)

        x = F.relu(self.conv_ch_1(x))
        x = F.relu(self.conv1(x))
        x = x.view(-1, math.ceil((((self.input_len-11)))*32))

        x = torch.cat([x, hidden], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, hidden


def load_database(data_path, data, cutoff, batch_size, pin_memory=False, test=False):
    data = CustomLoader(data_path, data, cutoff)
    if test:
        sampler = SequentialSampler(range(len(data)))
    else:
        sampler = SubsetRandomSampler(range(len(data)))
    loader = DataLoader(data,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=8,
                        collate_fn=default_collate,
                        pin_memory=pin_memory)

    return loader


def train_model(data_path, train_data, test_data, cutoff, test_cutoff, dest, batch_size,
                epochs, GPU):
    train_loader = load_database(data_path, train_data, cutoff, batch_size, GPU)
    print("{} samples in train data".format(len(train_loader.dataset.X_train)))
    test_loader = load_database(data_path, test_data, cutoff, batch_size, GPU, True)
    print("{} samples in test data".format(len(test_loader.dataset.X_train)))

    hidden_size = 128
    if GPU:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    ratio = sum(train_loader.dataset.y_train)/len(train_loader.dataset.y_train)
    weights = torch.FloatTensor([ratio, 1-ratio]).type(dtype)
    model = DualComplex(batch_size, hidden_size, 2, True)
    model.type(dtype)
    loss = nn.CrossEntropyLoss(weights)
    optimizer = Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = StepLR(optimizer, 11, gamma=0.1)
    test_keys = ["test_data"]
    log = logger(["loss", "AUC", "P-R", "acc"], False, test_keys)
    model.fit(train_loader, valid_loaders=[test_loader], valid_keys=test_keys,
              scheduler=scheduler, epochs=epochs, loss=loss,
              optimizer=optimizer, log=log, dest=dest, HYBRID=True, GPU=GPU)


def predict(data_path, test_data, test_cutoff, model_name, dest, batch_size, GPU):
    hidden_size = 128

    test_loader = load_database(data_path, test_data, test_cutoff, batch_size, True)
    model = DualComplex(batch_size, hidden_size, 2, True)

    model.load_state_dict(torch.load(model_name))
    pred, true = model.predict(test_loader, HYBRID=True, GPU=GPU)
    df_pred = extend_lib(test_loader.dataset.masked_list, pred)
    df_pred.to_csv(dest)


def execute_function(function, data_path, train_data, test_data, cutoff, test_cutoff, model_name, dest, batch_size,
                    epochs, GPU):
    if function == 'train':
        train_model(data_path, train_data, test_data, cutoff, test_cutoff, dest, batch_size,
                    epochs, GPU)
    else:
        predict(data_path, test_data, test_cutoff, model_name, dest, batch_size, GPU)


def main():
    parser = argparse.ArgumentParser(description="high-end script function for"
                                     "DeepRibo")
    parser.add_argument('function', type=str, choices=('train', 'predict'),
                        help="Train/Use model")
    parser.add_argument('-data_path', type=str,
                        help="Directory containing the training/testing data folders")
    parser.add_argument('-train_data', default='[]', nargs='+', type=str,
                        help="Train data folder names present in the data path")
    parser.add_argument('-test_data', nargs='+', type=str,
                        help="Test data folder names present in the data path")
    parser.add_argument('-train_cutoff_rpkm', '--cr', nargs='+', type=float, help="cutoff RPKM values for the"
                        "training data")
    parser.add_argument('-train_cutoff_coverage', '--cc', nargs='+', type=float, help="cutoff"
                        "coverage values for the training data")
    parser.add_argument('-test_cutoff_rpkm', '--tcr', nargs='+', type=float, help="cutoff RPKM values for the"
                        "testing data")
    parser.add_argument('-test_cutoff_coverage', '--tcc', nargs='+', type=float, help="cutoff"
                        "coverage values for the testing data")
    parser.add_argument('-test_cutoff', nargs='+', type=float,
                        help="cutoff values of test data or data used for predict")
    parser.add_argument('-model', default='default', type=str, help="path to the model used for predictions")
    parser.add_argument('-dest', default='pred', type=str, help="Destination path for the"
                        "trained models or predictions")
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help="determines batch size")
    parser.add_argument('-e', '--epochs', default=25, type=int,
                        help="amount of epochs to train")
    parser.add_argument('--GPU', action="store_true")
    args = parser.parse_args()
    print(args)
    execute_function(args.function, args.data_path, args.train_data, args.test_data,
                    (args.cr, args.cc), (args.tcr, args.tcc),
                    args.model, args.dest, args.batch_size, args.epochs, args.GPU)


if __name__ == "__main__":
    sys.exit(main())
