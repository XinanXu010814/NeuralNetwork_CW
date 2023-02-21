from torch.utils.data import Dataset
from torch import nn


class Normalizer(object):
    def __init__(self, data):
        if len(data.shape) == 1:
            self.mean = data.mean()
            self.std = data.std()
        elif len(data.shape) == 2:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def reverse(self, x):
        return x * self.std + self.mean


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    def _activation(self, act):
        if act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'leaky':
            return nn.LeakyReLU()
        else:
            raise Exception('Unsupported activation!')

    def __init__(self, input_size, output_size, ls=[92, 106], acts=['relu', 'relu'], p=0.115):
        super().__init__()
        layers = []
        in_size = input_size
        for l, a in zip(ls, acts):
            if l > 0:
              layers.append(nn.Linear(in_size, l).double())
              layers.append(self._activation(a))
              layers.append(nn.Dropout(p=p))
              in_size = l
        layers.append(nn.Linear(in_size, output_size).double())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def parse_file(p_train=0.9):
    import pandas as pd
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    test_index = int(data.shape[0] * p_train)
    x_train = data.loc[:test_index, data.columns != output_label]
    y_train = data.loc[:test_index, [output_label]]
    x_validation = data.loc[test_index+1:, data.columns != output_label]
    y_validation = data.loc[test_index+1:, [output_label]]

    if p_train == 1:
        return x_train, y_train
    elif p_train == 0:
        return x_validation, y_validation
    else:
        return x_train, y_train, x_validation, y_validation
