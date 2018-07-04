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
from functions import FitModule, defaultCollate, Logger, Adam, extendLib,\
       str2bool


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
        self.X_train = self.masked_list.loc[:, 'filename':'filename_counts']
        self.y_train = self.masked_list['label'].values

    def __getitem__(self, index):
        # load and transform DNA sequence data
        path = '{}/{}'.format(self.data_path, self.X_train.iloc[index, 0])
        img = torch.from_numpy(torch.load(path))
        path = "{}/{}".format(self.data_path, self.X_train.iloc[index, 1])
        counts = torch.from_numpy(torch.load(path))
        label = self.y_train[index]
        return img, counts, label

    def __len__(self):
        return len(self.X_train.index)

    def sortData(self):
        sort_idx = np.argsort(abs(self.masked_list['start_site'] -
                                  self.masked_list['stop_site']))
        self.X_train = self.X_train.iloc[sort_idx, :]
        self.y_train = self.y_train[sort_idx]
        self.masked_list = self.masked_list.iloc[sort_idx, :]

    def bucketshuffle(self, batch_size):
        self.sortData()
        # shuffle rows within regions
        index = np.array(self.masked_list.index)
        region_size = int(len(index)/batch_size//12)
        inc_batch_reg = len(index) % region_size
        index_list = np.array(index[inc_batch_reg:])
        np.random.shuffle(np.reshape(index_list, (-1, region_size)).T)
        shuffled_index = np.hstack((index_list, index[:inc_batch_reg]))

        # shuffle batches within dataset
        inc_batch = len(index) % batch_size
        shuffled_index_list = shuffled_index[inc_batch:]
        np.random.shuffle(np.reshape(shuffled_index_list, (-1, batch_size)))

        # set dataset with new order
        sort_idx = np.hstack((shuffled_index_list, shuffled_index[:inc_batch]))
        self.X_train = self.X_train.loc[sort_idx, :]
        self.y_train = self.y_train[sort_idx]
        self.masked_list = self.masked_list.loc[sort_idx, :]


class DualComplex(FitModule):
    def __init__(self, motif_count, hidden_size, layers, bidirect, nodes):
        """The DeepRibo model architecture

        Arguments:
            hidden_size (int): weights allocated to the GRU
            layers (int): amount of GRU layers
            bidirect (bool): model uses a bidirectional GRU
        """
        super(DualComplex, self).__init__()
        self.gru = nn.GRU(1, hidden_size=hidden_size,
                          num_layers=layers, dropout=0.3, bidirectional=bidirect)
        self.layers = layers
        self.hidden_size = hidden_size
        self.motif_count = motif_count
        self.in_len = 30
        self.bi = 2**bidirect
        self.nodes_0 = self.hidden_size*layers*self.bi + (self.in_len-11)*self.motif_count
        nodes.append(2)
        nodes.insert(0, self.nodes_0)
        self.conv_ch_1 = nn.Conv2d(4, 4, (1, 1))
        self.conv1 = nn.Conv2d(4, self.motif_count, (12, 1))
        fc = []
        for i in range(len(nodes)-2):
            fc.append(nn.Linear(nodes[i], nodes[i+1]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(nodes[-2], nodes[-1]))
        self.fc = nn.Sequential(*fc)

    def forward(self, x, hidden=None):
        # the input x is a tuple containing the DNA sequence data ([0])
        # and RIBO-seq data ([1])
        y = x[1]
        y, hidden = self.gru(y)
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(-1, self.hidden_size*self.layers*self.bi).contiguous()
        x = x[0]

        x = F.relu(self.conv_ch_1(x))
        x = F.relu(self.conv1(x))
        x = x.view(-1, (self.in_len-11)*self.motif_count)
        x = torch.cat([x, hidden], dim=1).contiguous()
        x = self.fc(x)

        return x, hidden


class CNNComplex(FitModule):
    def __init__(self, motif_count, nodes):
        """The DeepRibo model architecture

        Arguments:
            hidden_size (int): weights allocated to the GRU
            layers (int): amount of GRU layers
            bidirect (bool): model uses a bidirectional GRU
        """
        super(CNNComplex, self).__init__()
        self.motif_count = motif_count
        self.in_len = 30
        self.nodes_0 = (self.in_len-11)*self.motif_count
        nodes.append(2)
        nodes.insert(0, self.nodes_0)
        self.conv_ch_1 = nn.Conv2d(4, 4, (1, 1))
        self.conv1 = nn.Conv2d(4, self.motif_count, (12, 1))
        fc = []
        for i in range(len(nodes)-2):
            fc.append(nn.Linear(nodes[i], nodes[i+1]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(nodes[-2], nodes[-1]))
        self.fc = nn.Sequential(*fc)

    def forward(self, x, hidden=None):
        x = F.relu(self.conv_ch_1(x[0]))
        x = F.relu(self.conv1(x))
        x = x.view(-1, (self.in_len-11)*self.motif_count)
        x = self.fc(x)

        return x, hidden


class RNNComplex(FitModule):
    def __init__(self, hidden_size, layers, bidirect, nodes):
        """The DeepRibo model architecture

        Arguments:
            hidden_size (int): weights allocated to the GRU
            layers (int): amount of GRU layers
            bidirect (bool): model uses a bidirectional GRU
        """
        super(RNNComplex, self).__init__()
        self.gru = nn.GRU(1, hidden_size=hidden_size,
                          num_layers=layers, dropout=0.3, bidirectional=bidirect)
        self.layers = layers
        self.hidden_size = hidden_size
        self.in_len = 30
        self.bi = 2**bidirect
        self.nodes_0 = self.hidden_size*layers*self.bi
        nodes.append(2)
        nodes.insert(0, self.nodes_0)
        fc = []
        for i in range(len(nodes)-2):
            fc.append(nn.Linear(nodes[i], nodes[i+1]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(nodes[-2], nodes[-1]))
        self.fc = nn.Sequential(*fc)

    def forward(self, x, hidden=None):
        y = x[1]
        y, hidden = self.gru(y)
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(-1, self.hidden_size*self.layers*self.bi).contiguous()

        x = self.fc(hidden)

        return x, hidden


def loadDatabase(data_path, data, cutoff, batch_size, num_workers, pin_memory,
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
    data.sortData()
    idx = np.arange(len(data.masked_list))
    dfs = data.masked_list.iloc[:, 0].str.split('/').str[0].value_counts()
    labels = np.hstack([np.full(x, i) for i, x in enumerate(dfs.values)])
    if valid_size != 0:
        train_idx, valid_idx = train_test_split(idx, test_size=valid_size,
                                                stratify=labels)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_sampler = SequentialSampler(train_idx)
        #train_sampler = SubsetRandomSampler(train_idx)
        valid_loader = DataLoader(data,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=num_workers,
                                  collate_fn=defaultCollate,
                                  pin_memory=pin_memory)
        train_loader = DataLoader(data,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=defaultCollate,
                                  pin_memory=pin_memory)

        return train_loader, valid_loader

    else:
        train_sampler = SubsetRandomSampler(idx)
        train_loader = DataLoader(data,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=defaultCollate,
                                  pin_memory=pin_memory)

        return train_loader


def trainModel(args, data_path, train_data, valid_size, train_cutoff,
               dest, batch_size, epochs, hidden_size, layers, bidirect, motif_count, nodes,
               model_type, num_workers, GPU, verbose):
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
                                              batch_size, num_workers, GPU, valid_size)
    print("{} samples in train data".format(len(train_loader.sampler)))
    print("{} samples in valid data".format(len(valid_loader.sampler)))

    if GPU:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # create weighted loss (heavily imbalanced data)
    ratio = sum(train_loader.dataset.y_train)/len(train_loader.dataset.y_train)
    weights = torch.FloatTensor([ratio, 1-ratio]).to(device)
    # initialize model
    if model_type == 'CNNRNN':
        model = DualComplex(motif_count, hidden_size, layers, bidirect, nodes)
    if model_type == 'CNN':
        model = CNNComplex(motif_count, nodes)
    else:
        model = RNNComplex(hidden_size, layers, bidirect, nodes)
    model.to(device)
    # nn.DataParallel(model, device_ids=[2, 0])
    loss = nn.CrossEntropyLoss(weights)
    optimizer = Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = StepLR(optimizer, 10, gamma=0.1)
    # key under which performance measures are saved
    valid_keys = ["valid_data"]
    # record loss, accuracy, AUC, PR-AUC on test set
    log = Logger(vars(args), ["loss", "AUC", "P-R", "acc"], False, valid_keys)
    # train the model
    model.fit(device, train_loader, valid_loaders=[valid_loader], valid_keys=valid_keys,
              scheduler=scheduler, epochs=epochs, loss=loss,
              optimizer=optimizer, log=log, dest=dest, GPU=GPU, verbose=verbose)


def predict(data_path, pred_data, pred_cutoff, model_name, dest, batch_size,
            hidden_size, layers, bidirect, motif_count, nodes, model_type,
            num_workers, GPU, verbose):
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
    pred_loader = loadDatabase(data_path, pred_data, pred_cutoff, batch_size,
                               num_workers, GPU, verbose)
    if model_type == 'CNNRNN':
        model = DualComplex(motif_count, hidden_size, layers, bidirect, nodes)
    if model_type == 'CNN':
        model = CNNComplex(motif_count, nodes)
    else:
        model = RNNComplex(hidden_size, layers, bidirect, nodes)

    model.load_state_dict(torch.load(model_name))
    pred, true = model.predict(pred_loader, GPU=GPU, verbose=verbose)
    df_pred = extendLib(pred_loader.dataset.masked_list, pred)
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
                       description='Train a model',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            # TWO argvs, ie the command (git) and the subcommand (commit)
            parser.add_argument('data_path', type=str,
                                help="path containing the data folders for "
                                "training and testing")
            parser.add_argument('--train_data', default='[]', nargs='+',
                                type=str, required=True, help="train data "
                                "folder names present in the data path")
            parser.add_argument('--valid_size', type=float,
                                default=0.05,
                                help="percentage of train used as valid"
                                "data")
            parser.add_argument('-r', '--rpkm', nargs='+', type=float,
                                required=True,
                                help="minimum cutoff of RPKM values to filter "
                                "the training data")
            parser.add_argument('-c', '--coverage', nargs='+', type=float,
                                required=True, help="minimum cutoff of"
                                "coverage values to filter the training data"
                                ", these are given in the same order.")
            parser.add_argument('-d', '--dest', default='pred', type=str,
                                help="path to which the model is saved")
            parser.add_argument('-b', '--batch_size', type=int, default=256,
                                help="training batch size")
            parser.add_argument('-e', '--epochs', default=20, type=int,
                                help="training epochs")
            parser.add_argument('-g', '--GRU_nodes', default=128, type=int,
                                help="size of the hidden state of the GRU "
                                "unit")
            parser.add_argument('-l', '--GRU_layers', default=2,
                                choices=[1, 2], type=int, help="amount of "
                                "sequential GRU layers")
            parser.add_argument('-B', '--GRU_bidirect', type=str2bool, nargs='?',
                                const=True, default=True, help="use of "
                                "bidirectional GRU units")
            parser.add_argument('-m', '--COV_motifs', default=32, type=int,
                                help="amount of motifs (conv kernels) used "
                                "by the convolutional layer")
            parser.add_argument('-n', '--FC_nodes', default=[1024, 512],
                                type=int, nargs='+', help="nodes per layer "
                                "present in the fully connected layers of "
                                "DeepRibo")
            parser.add_argument('--model_type', default='CNNRNN', type=str,
                                choices=['CNNRNN', 'CNN', 'RNN'], help=""
                                "Use CNNRNN, CNN or RNN architecture")
            parser.add_argument('--num_workers', default=0, type=int,
                                help="numbers of CPU units used for data"
                                "loading")
            parser.add_argument('--GPU', action='store_true', help=""
                                "use of GPU (RECOMMENDED)")
            parser.add_argument('-v', '--verbose', action='store_true', help=""
                                "more detailed progress bar")
            args = parser.parse_args(sys.argv[2:])
            print('Training a model with parameters: {}'.format(args))
            trainModel(args, args.data_path, args.train_data, args.valid_size,
                       (args.rpkm, args.coverage), args.dest,
                       args.batch_size, args.epochs, args.GRU_nodes,
                       args.GRU_layers, args.GRU_bidirect, args.COV_motifs,
                       args.FC_nodes, args.model_type, args.num_workers,
                       args.GPU, args.verbose)

        def predict(self):
            parser = argparse.ArgumentParser(
                        description='Create predictions using a trained model',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser.add_argument('data_path', type=str,
                                help="path containing the data folders for"
                                " predictions")
            parser.add_argument('--pred_data', type=str, required=True,
                                help="data folder name present in the data "
                                "path used to make predictions on")
            parser.add_argument('-r', '--rpkm',  type=float,
                                required=True, help="minimum cutoff of RPKM "
                                "value to filter the data used for "
                                "predictions.")
            parser.add_argument('-c', '--coverage', nargs='+', type=float,
                                required=True, help="minimum cutoff of "
                                "coverage value to filter the data used for "
                                "predictions order")
            parser.add_argument('-M', '--model', type=str, required=True,
                                help="path to the trained model")
            parser.add_argument('-d', '--dest', default='pred', type=str,
                                required=True, help="path to file in which "
                                "predictions are saved")
            parser.add_argument('-g', '--GRU_nodes', default=128, type=int,
                                help="size of the hidden state of the GRU "
                                "unit")
            parser.add_argument('-l', '--GRU_layers', default=2, choices=[1, 2],
                                type=str, help="amount of sequential GRU "
                                "layers")
            parser.add_argument('-B', '--GRU_bidirect', default=True,
                                type=bool, help="use of bidirectional GRU "
                                "units")
            parser.add_argument('-m', '--COV_motifs', default=32, type=int,
                                help="amount of motifs (conv kernels) used "
                                "by the convolutional layer")
            parser.add_argument('-n', '--FC_nodes', default=[1024, 512],
                                type=int, nargs='+', help="nodes per layer "
                                "present in the fully connected layers of "
                                "DeepRibo")
            parser.add_argument('--model_type', default='CNNRNN', type=str,
                                choices=['CNNRNN', 'CNN', 'RNN'], help=""
                                "Use CNNRNN, CNN or RNN architecture")
            parser.add_argument('--num_workers', default=0, type=int,
                                help="numbers of CPU units used for data"
                                "loading")
            parser.add_argument('--GPU', action='store_true', help=""
                                "use of GPU")
            parser.add_argument('-v', '--verbose', action='store_true', help=""
                                "more detailed progress bar")
            args = parser.parse_args(sys.argv[2:])
            print('Creating predictions using model {}'.format(args.model))
            predict(args.data_path, [args.pred_data],
                    ([args.rpkm], [args.coverage]), args.model,
                    args.dest, 1024, args.GRU_nodes, args.GRU_layers,
                    args.GRU_bidirect, args.COV_motifs, args.FC_nodes,
                    args.model_type, args.num_workers, args.GPU,
                    args.verbose)


if __name__ == "__main__":
    sys.exit(ParseArgs())
