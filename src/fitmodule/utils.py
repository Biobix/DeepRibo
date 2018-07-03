import numpy as np
import sys
import torch

from functools import partial
from torch.utils.data import DataLoader, TensorDataset


##### Data utils #####

def get_loader(X, y=None, batch_size=1, shuffle=False):
    """Convert X and y Tensors to a DataLoader
        If y is None, use a dummy Tensor
    """
    if y is None:
        y = torch.Tensor(X.size()[0])
    return DataLoader(TensorDataset(X, y), batch_size, shuffle)


##### Logging #####

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
                if i == self.ticks[-1]:
                    sys.stdout.write("\n")
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n-1)
        sys.stdout.write("{0}\n\n".format(message))
        sys.stdout.flush()
