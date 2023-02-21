import torch
import pickle
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import torch.optim as optim
from torch import nn
from utils import *


class Regressor():

    def __init__(self, x, nb_epoch=100, ls=[92, 106], acts=['relu', 'relu'], dropout_rate=0.115):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        # Create neural network
        self.network = NeuralNetwork(self.input_size, self.output_size,
                                     ls=ls, acts=acts, p=dropout_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False, normalize_y=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        # Use a copy to avoid modification on original array
        x = x.copy()

        # Perform one-hot encoding
        if training and "ocean_proximity" in x.columns.values:
            from sklearn import preprocessing
            self._lb = preprocessing.LabelBinarizer()
            self._lb.fit(x["ocean_proximity"])
        if "ocean_proximity" in x.columns.values:
            transformed_cols = self._lb.transform(x["ocean_proximity"])
            x.drop(columns='ocean_proximity', inplace=True)
            for i in range(len(self._lb.classes_)):
                x.insert(0, self._lb.classes_[i], transformed_cols[:, i])

        # Fill holes in data
        x.fillna(x.median(), inplace=True)

        list_attr = ["total_rooms", "total_bedrooms", "population"]
        if "households" in x.columns.values:
            for col in list_attr:
                if col in x.columns.values:
                    # avoid divide by 0 error
                    x[col + "_per_household"] = x[col] / (x["households"] + 1)
                    x = x.drop(columns=col)

        list_log = ["housing_median_age", "median_income", "total_rooms_per_household",
                    "total_bedrooms_per_household", "population_per_household"]
        for col in list_log:
            # plus 1 to avoid 0 value inside log
            if col in x.columns.values:
                x[col] = np.log(x[col] + 1)
        if training and y is not None:
            y = np.log(y + 1)

        # Standardize data
        if training:
            self.x_normalizer = Normalizer(x.to_numpy())
        x = self.x_normalizer.normalize(x.to_numpy())
        if y is not None:
            if training:
                self.y_normalizer = Normalizer(y.to_numpy())
                y = self.y_normalizer.normalize(y.to_numpy())
            else:
                if normalize_y:
                    y = self.y_normalizer.normalize(y.to_numpy())
                else:
                    y = y.to_numpy()
        return torch.tensor(x, dtype=float), (torch.tensor(y, dtype=float) if y is not None else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y, batch_size=64, lr=0.001, print_loss=True):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training=True)
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        dataset = NumpyDataset(X, Y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for i in range(self.nb_epoch):
            train(self.network, data_loader, loss_fn, optimizer)
            with torch.no_grad():
                loss = loss_fn(self.network(X), Y)
                if print_loss:
                    print(f"loss: {loss.item():>7f}  [{i+1:>5d}/{self.nb_epoch:>5d}]")
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.network.eval()
        x, _ = self._preprocessor(x, training=False)
        if self.y_normalizer is None:
            print("Predict is called when model is not trained!")
            with torch.no_grad():
                return np.exp(self.network(x).detach().numpy())
        with torch.no_grad():
            y = self.y_normalizer.reverse(self.network(x).detach().numpy())
        return np.exp(y)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Y is converted to torch tensor but not standardized 
        _, Y = self._preprocessor(x, y, training=False)
        pred = self.predict(x)
        actual = Y.detach().numpy()
        print("R^2 : ", r2_score(actual, pred))
        print("MAE : ", mean_absolute_error(actual, pred))
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        print("MSE : ", mse)
        print("RMSE : ", rmse)
        return rmse

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

# used in hyperparameter training
def test(model, data_loader):
    model.eval()
    res = []
    with torch.no_grad():
        for (data, target) in data_loader:
            outputs = model(data)
            res.append(mean_squared_error(outputs, target))

    return sum(res) / len(res)

# used in hyperparameter training
def train(model, data_loader, loss_fn, optimizer):
    model.train()
    for i, (x, y) in enumerate(data_loader):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    print("starting to save regressor")
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, y_train, x_valid, y_valid, valid_rmse_weight=2, phase_I=False):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        If phase_I is True, phase I random search described in the report will be used; otherwise, phase II
        grid search combined with random search will be used

    Returns:
        a dictionary with keys:
            rmse                     : float64
            time_this_iter_s         : float64
            done                     : bool
            timesteps_total          : float64
            episodes_total           : float64
            training_iteration       : int64
            trial_id                 : object
            experiment_id            : object
            date                     : object
            timestamp                : int64
            time_total_s             : float64
            pid                      : int64
            hostname                 : object
            node_ip                  : object
            time_since_restore       : float64
            timesteps_since_restore  : int64
            iterations_since_restore : int64
            warmup_time              : float64
            config/acts              : object
            config/batch_size        : int64
            config/layer_range       : object
            config/lr                : float64
            logdir                   : object
    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    import os
    from ray import air, tune
    from ray.air import session
    from ray.air.checkpoint import Checkpoint
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from itertools import product

    num_layers = 2
    layer_tuples = [[]]
    for i in range(num_layers):
        layer_tuples += list(product(range(10, 101, 10), repeat=num_layers-i))
    acts = list(product(["relu", "tanh", "leaky"], repeat=num_layers))
    # configuration for phase II search; grid search is used in combination with random search
    config = {
        "batch_size": tune.choice([32, 64, 128]),
        "layer_range": tune.grid_search(layer_tuples),
        "l1d": tune.randint(0, 10),
        "l2d": tune.randint(0, 10),
        "lr": tune.uniform(0.001, 0.005),
        "acts": tune.grid_search(acts),
        "dropout_rate": tune.uniform(0.1, 0.5)
    }
    epochs = 100
    if phase_I:
        # configuration for phase I search; random search is used
        acts = list(product(["relu", "tanh", "leaky", "sigmoid"], repeat=num_layers))
        config = {
            "epochs": tune.randint(10, 201),
            "batch_size": tune.choice([16, 32, 64, 128, 256, 512]),
            "l1": tune.randint(0, 201),
            "l2": tune.randint(0, 201),
            "lr": tune.uniform(0.0001, 0.1),
            "acts": tune.choice(acts),
            "dropout_rate": tune.uniform(0, 0.9)
        }

    def train_tune(config):
        train_loader = DataLoader(NumpyDataset(x_train, y_train),
                                  batch_size=config["batch_size"], shuffle=True)
        valid_loader = DataLoader(NumpyDataset(x_valid, y_valid),
                                  batch_size=config["batch_size"], shuffle=True)
        if phase_I:
            layers = [config["l1"], config["l2"]]
        else:
            layers = []
            for r, d in zip(config["layer_range"], [config["l1d"], config["l2d"]]):
                layers.append(r+d)
        model = NeuralNetwork(x_train.shape[1],
                              1,
                              ls=layers,
                              acts=config["acts"],
                              p=config["dropout_rate"])
        optimizer = optim.Adam(
            model.parameters(), lr=config["lr"])

        # epochs only investigate as a hyperparameter in phase I
        if phase_I:
            _epochs = config["epochs"]
        else:
            _epochs = epochs
        for i in range(_epochs):
            train(model, train_loader, nn.MSELoss(), optimizer)
            rmse_train = test(model, train_loader)
            rmse_valid = test(model, valid_loader)
            if i == epochs - 1:
                os.makedirs("~/raytune", exist_ok=True)
                torch.save(
                    (model.state_dict(), optimizer.state_dict()),
                    os.path.join("~/raytune/checkpoint.pt"),
                )
                checkpoint = Checkpoint.from_directory("~/raytune")
                session.report({"mse": rmse_train + valid_rmse_weight * rmse_valid,
                               "done": True}, checkpoint=checkpoint)
            else:
                session.report({"mse": rmse_train + valid_rmse_weight * rmse_valid})

    # ASHA Scheduler mentioned in the report
    scheduler = ASHAScheduler(
        metric="mse",
        mode="min",
        max_t=epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=["layer_range", "lr", "batch_size"],
        metric_columns=["mse", "training_iteration"],
    )

    tuner = tune.Tuner(
        train_tune,
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=5 if not phase_I else 20,
        ),
        run_config=air.RunConfig(
            name="tune",
            progress_reporter=reporter,
            verbose=1,
        ),
    )
    results = tuner.fit()

    data_frame = results.get_dataframe(filter_metric="mse", filter_mode="min")
    best_frame = data_frame.sort_values(by="mse")
    # return the top 3 configurations
    return best_frame.to_dict('records')[:3]

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def parse_file():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    # change it to absolute path on ur PC
    data = pd.read_csv("./housing_test.csv")

    # Splitting input and output
    train_index = int(data.shape[0] * 0.8)
    x_train = data.loc[:train_index, data.columns != output_label]
    y_train = data.loc[:train_index, [output_label]]

    valid_index = train_index + int(data.shape[0] * 0.1)
    x_validation = data.loc[train_index+1:valid_index, data.columns != output_label]
    y_validation = data.loc[train_index+1:valid_index, [output_label]]
    x_test = data.loc[valid_index+1:, data.columns != output_label]
    y_test = data.loc[valid_index+1:, [output_label]]

    return (x_train, y_train, x_validation, y_validation, x_test, y_test)


def example_main():

    x_train, y_train, x_validation, y_validation, x_test, y_test = parse_file()

    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=100)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # train test
    print("---train test---")
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

    # validation test
    print("---validation test---")
    error = regressor.score(x_validation, y_validation)
    print("\nRegressor error: {}\n".format(error))

    # final test
    print("---final test---")
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
