import math
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from functions import FitModule, default_collate, Logger, Adam, extend_lib


class CustomLoader(Dataset):
    def __init__(self, data_path, data, cutoff):
        """A custom data loader for PyTorch

        Arguments:
            data_path (string): path to main directory containing all
                experimental data
            data (list): list of folder names in data_path which are connected
                in one dataset
            cutoff (tupel): Tupel containing two lists each listing the
                minimum RPKM ([0]) and coverage ([1]) cutoff values for each of
                the datasets in data. Values given must be in the same
                sequential order as data.
        """
        self.data_path = data_path
        data_list = ["{}/{}/data_list.csv".format(data_path, d) for d in data]
        print(data_list)
        # load all data in dataframe
        tmp_df = pd.concat([pd.read_csv(dl) for dl in data_list])
        self.list = tmp_df

        # 1. mask data in each dataset which does not fullfill its own
        # constraints for both the minimum RPKM as coverage
        mask = np.array([])
        amask = np.array([])
        for x in zip(data_list, cutoff[0], cutoff[1]):
            tmp_s_df = pd.read_csv(x[0])
            cond_1 = tmp_s_df["rpk_elo"] >= x[1]
            cond_2 = tmp_s_df["coverage_elo"] >= x[2]
            mask = np.hstack((mask, np.logical_and(cond_1, cond_2)))
            amask = np.hstack((amask, np.logical_or(cond_1 == False,
                                                    cond_2 == False)))
        self.cond_1 = mask
        self.anti_cond_1 = amask

        # 2. mask data in collected data which is smaller than 30 
        # nucleotides
        self.cond_2 = abs(tmp_df["start_site"]-tmp_df["stop_site"]) > 30
        self.mask = np.logical_and(self.cond_1, self.cond_2)
        self.anti_mask = np.logical_and(self.anti_cond_1, self.cond_2)

        # 3. filter out ORFs sharing a stop codon with an annotated
        # ORF which has been filtered out by previous constraints  
        self.masked_list = tmp_df[self.mask].reset_index(drop=True)
        list_neg = tmp_df[self.anti_mask][tmp_df[self.anti_mask]["label"] == 1]

        con_1 = tmp_df["strand"].isin(list_neg["strand"])
        con_2 = tmp_df["stop_site"].isin(list_neg["stop_site"])
        temp_mask = np.logical_and(con_1, con_2) == False
        self.mask = np.logical_and(self.mask, temp_mask)
        self.masked_list = tmp_df[self.mask].reset_index(drop=True)

        # create a dataframe containing the locations of the input 
        # data for each sample
        X_train = tmp_df.loc[:, 'filename':'filename_counts'][self.mask]
        self.X_train = X_train.reset_index(drop=True)
        self.y_train = tmp_df['label'][self.mask].values

    def __getitem__(self, index):
        # load and transform DNA sequence data
        path = '{}/{}'.format(self.data_path, self.X_train.iloc[index, 0])
        img = torch.from_numpy(np.load(path)).contiguous()
        img = img.view(1, img.shape[0],
                       img.shape[1]).transpose(0, 2)
        # load and transform RIBO-seq sequence data
        path = "{}/{}".format(self.data_path, self.X_train.iloc[index, 1])
        counts = torch.from_numpy(np.load(path))
        label = self.y_train[index]
        return img, counts, label

    def __len__(self):
        return len(self.X_train.index)


class DualComplex(FitModule):
    def __init__(self, hidden_size, layers, bidirect=False):
        """The DeepRibo model architecture

        Arguments:
            hidden_size (int): weights allocated to the GRU
            layers (int): amount of GRU layers
            bidirect (bool): model uses a bidirectional GRU
        """
        super(DualComplex, self).__init__()
        self.gru = nn.GRU(1, hidden_size=hidden_size,
                          num_layers=layers, bidirectional=bidirect)
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
        # the input x is a tuple containing the DNA sequence data ([0])
        # and RIBO-seq data ([1])
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


def loadDatabase(data_path, data, cutoff, batch_size, pin_memory=False,
                 valid_size=0):
    """Loads data using a custom loader

    Arguments:
        data_path (string): path to main directory containing all experimental
            data
        data (list): list of folder names in data_path which are connected in
            one dataset
        cutoff (tupel): Tupel containing two lists each listing the
            minimum RPKM ([0]) and coverage ([1]) cutoff values for each of
            the datasets in data. Values given must be in the same
            sequential order as data.
        batch_size (int): batch size (default:32)
        pin_memory (bool): The use of allocated GPU memory for faster
            processing (default: False)
        test (bool): data used for testing (default: False)

    """
    data = CustomLoader(data_path, data, cutoff)
    idx = np.arange(len(data.masked_list))
    dfs = data.masked_list.iloc[:, 0].str.split('/').str[0].value_counts()
    labels = np.hstack([np.full(x, i) for i, x in enumerate(dfs.values)])
    if valid_size != 0:
        train_idx, valid_idx = train_test_split(idx, test_size=valid_size,
                                                stratify=labels)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_loader = DataLoader(data,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=0,
                                  collate_fn=default_collate,
                                  pin_memory=pin_memory)
        train_loader = DataLoader(data,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=0,
                                  collate_fn=default_collate,
                                  pin_memory=pin_memory)

        return train_loader, valid_loader

    else:
        train_sampler = SubsetRandomSampler(idx)
        train_loader = DataLoader(data,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=0,
                                  collate_fn=default_collate,
                                  pin_memory=pin_memory)

        return train_loader


def trainModel(data_path, train_data, valid_size, train_cutoff, test_cutoff,
               dest, batch_size, epochs, GPU):
    """Trains the model using DeepRibo methodology

    Arguments:
        data_path (string): path to main directory containing all experimental
            data
        train_data (list): list of folder names in data_path used for training
        test_data (list): list of folder names in data_path used for testing
        train_cutoff (tupel): Tupel containing two lists each listing the
            minimum RPKM ([0]) and coverage ([1]) cutoff values for each of
            the datasets in train_data. Values given must be in the same
            sequential order as train_data.
        test_cutoff (tupel): Tupel containing two lists each listing the
            minimum RPKM ([0]) and coverage ([1]) cutoff values for each of
            the datasets in test_data. Values given must be in the same
            sequential order as test_data.
        dest (string): path to folder in which the model is saved
        batch_size (int): batch size (default:32)
        epochs (int): training epochs (default:25)
        GPU (bool): trains model using a GPU
    """
    train_loader, valid_loader = loadDatabase(data_path, train_data, train_cutoff,
                                              batch_size, GPU, valid_size)
    print("{} samples in train data".format(len(train_loader.sampler)))
    print("{} samples in valid data".format(len(valid_loader.sampler)))

    hidden_size = 128

    if GPU:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # create weighted loss (heavily imbalanced data)
    ratio = sum(train_loader.dataset.y_train)/len(train_loader.dataset.y_train)
    weights = torch.FloatTensor([ratio, 1-ratio]).type(dtype)
    # initialize model
    model = DualComplex(hidden_size, 2, True)
    model.type(dtype)
    loss = nn.CrossEntropyLoss(weights)
    optimizer = Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = StepLR(optimizer, 10, gamma=0.1)
    # key under which performance measures are saved
    test_keys = ["test_data"]
    # record loss, accuracy, AUC, PR-AUC on test set
    log = Logger(["loss", "AUC", "P-R", "acc"], False, test_keys)
    # train the model
    model.fit(train_loader, valid_loaders=[valid_loader], valid_keys=test_keys,
              scheduler=scheduler, epochs=epochs, loss=loss,
              optimizer=optimizer, log=log, dest=dest, GPU=GPU)


def predict(data_path, pred_data, pred_cutoff, model_name, dest, batch_size,
            GPU):
    """Uses the model for predictions

    Arguments:
        data_path (string): path to main directory containing experimental
            data
        test_data (list): list of folder names in data_path used for pred-
            ictions
        train_cutoff (tupel): Tupel containing two lists each listing the
            minimum RPKM ([0]) and coverage ([1]) cutoff values for each of
            the datasets in train_data. Values given must be in the same
            sequential order as train_data.
        test_cutoff (tupel): Tupel containing two lists each listing the
            minimum RPKM ([0]) and coverage ([1]) cutoff values for each of
            the datasets in test_data. Values given must be in the same
            sequential order as test_data.
        dest (string): path to folder in which the model is saved
        batch_size (int): batch size (default:32)
        epochs (int): training epochs (default:25)
        GPU (bool): trains model using a GPU
    """
    hidden_size = 128

    pred_loader = loadDatabase(data_path, pred_data, pred_cutoff, batch_size,
                               True)
    model = DualComplex(hidden_size, 2, True)

    model.load_state_dict(torch.load(model_name))
    pred, true = model.predict(pred_loader, GPU=GPU)
    df_pred = extend_lib(pred_loader.dataset.masked_list, pred)
    df_pred.to_csv(dest)


class ParseArgs(object):

        def __init__(self):
            parser = argparse.ArgumentParser(
                        description='Tool for training/using models',
                        usage='''DeepRibo.py <command> [<args>]

             Commands:
               train     Train a model using parsed data
               predict   Make predictions using a trained model
            ''')
            parser.add_argument('command', help='Subcommand to run')
            args = parser.parse_args(sys.argv[1:2])
            if not hasattr(self, args.command):
                print('Unrecognized command')
                parser.print_help()
                exit(1)
            # use dispatch pattern to invoke method with same name
            getattr(self, args.command)()

        def train(self):
            parser = argparse.ArgumentParser(
                            description='Train a model')
            # TWO argvs, ie the command (git) and the subcommand (commit)
            parser.add_argument('data_path', type=str,
                                help="path containing the data folders for "
                                "training and testing")
            parser.add_argument('--train_data', default='[]', nargs='+',
                                type=str, required=True, help="train data "
                                "folder names present in the data path")
            parser.add_argument('--valid_size', nargs='*', type=int, default=0.05,
                                help="percentage of train used as valid"
                                "data")
            parser.add_argument('-r', '--tr_rpkm', nargs='+', type=float,
                                required=True,
                                help="minimum cutoff of RPKM values to filter "
                                "the training data")
            parser.add_argument('-c', '--tr_cov', nargs='+', type=float,
                                required=True, help="minimum cutoff of"
                                "coverage values to filter the training data"
                                ", these are given in the same order.")
            parser.add_argument('-tr', '--te_rpkm', nargs='*', type=float,
                                help="minimum cutoff of RPKM values to filter "
                                "the testing data"
                                ", these are given in the same order.")
            parser.add_argument('-tc', '--te_cov', nargs='*', type=float,
                                help="minimum cutoff of coverage values to "
                                "filter the testing data"
                                ", these are given in the same order.")
            parser.add_argument('-d', '--dest', default='pred', type=str,
                                help="path to which the model is saved")
            parser.add_argument('-b', '--batch_size', type=int, default=32,
                                help="training batch size")
            parser.add_argument('-e', '--epochs', default=20, type=int,
                                help="training epochs")
            parser.add_argument('--GPU', action="store_true", help=""
                                "use of GPU (RECOMMENDED)")
            args = parser.parse_args(sys.argv[2:])
            print('Training a model with parameters: {}'.format(args))
            trainModel(args.data_path, args.train_data, args.valid_size,
                       (args.tr_rpkm, args.tr_cov),
                       (args.te_rpkm, args.te_cov), args.dest,
                       args.batch_size, args.epochs, args.GPU)

        def predict(self):
            parser = argparse.ArgumentParser(
                        description='Create predictions using a trained model')
            parser.add_argument('data_path', type=str,
                                help="path containing the data folders for"
                                " predictions")
            parser.add_argument('--pred_data', type=str, required=True,
                                help="data folder name present in the data "
                                "path used to make predictions on")
            parser.add_argument('-pr', '--pr_rpkm',  type=float,
                                required=True, help="minimum cutoff of RPKM "
                                "value to filter the data used for "
                                "predictions.")
            parser.add_argument('-pc', '--pr_cov', nargs='+', type=float,
                                required=True, help="minimum cutoff of "
                                "coverage value to filter the data used for "
                                "predictions order")
            parser.add_argument('-m', '--model', type=str, required=True,
                                help="path to the trained model")
            parser.add_argument('-d', '--dest', default='pred', type=str,
                                required=True, help="path to file in which "
                                "predictions are saved")
            parser.add_argument('--GPU', action="store_true",
                                help="use of GPU")
            args = parser.parse_args(sys.argv[2:])
            print('Creating predictions using model {}'.format(args.model))
            predict(args.data_path, [args.pred_data],
                    ([args.pr_rpkm], [args.pr_cov]), args.model,
                    args.dest, 32, args.GPU)


if __name__ == "__main__":
    sys.exit(ParseArgs())
