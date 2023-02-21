from part2_house_value_regression import *
import os
from contextlib import redirect_stdout


def ray_tuning(id=-1, valid_rmse_weight=2):
    print(f"\n\n>>>>>>>>>>>>>>>>>>>> ray_tuning {id} <<<<<<<<<<<<<<<<<<<<\n\n")
    output_label = "median_house_value"

    if os.path.exists("./housing.csv"):
        data = pd.read_csv("./housing.csv")
    else:
        data = pd.read_csv("~/housing.csv")

    # Splitting input and output
    train_index = int(data.shape[0] * 0.6)
    valid_index = int(data.shape[0] * 0.8)
    x_train_raw = data.loc[:train_index, data.columns != output_label]
    y_train_raw = data.loc[:train_index, [output_label]]

    x_valid_raw = data.loc[train_index+1:valid_index, data.columns != output_label]
    y_valid_raw = data.loc[train_index+1:valid_index, [output_label]]

    x_test_raw = data.loc[valid_index+1:, data.columns != output_label]
    y_test_raw = data.loc[valid_index+1:, [output_label]]

    regressor = Regressor(x_train_raw)
    x_train, y_train = regressor._preprocessor(x_train_raw, y_train_raw, True)
    x_valid, y_valid = regressor._preprocessor(x_valid_raw, y_valid_raw, False, True)
    best_results = RegressorHyperParameterSearch(
        x_train, y_train, x_valid, y_valid, valid_rmse_weight=valid_rmse_weight)
    def print_report():
        for i, best_result in enumerate(best_results):
            print(f"\n\n\n========== Report {i} ==========")
            for key, value in best_result.items():
                print(">>>", key, ":", value)
    print_report()
    with open("report.log", 'w') as f:
        with redirect_stdout(f):
            print_report()
    
    ret_val = None
    with open("config_and_test.log", 'w') as f:
        with redirect_stdout(f):
            print("")
    
    for i, br in enumerate(best_results):
        best_result = br

        epochs = best_result["training_iteration"]
        acts = best_result["config/acts"]
        batch_size = best_result["config/batch_size"]
        layer_size = best_result["config/layer_range"]
        l1d = best_result["config/l1d"]
        l2d = best_result["config/l2d"]
        layer_size = [a + b for a, b in zip(layer_size, [l1d, l2d])]
        lr = best_result["config/lr"]
        dropout_rate = best_result["config/dropout_rate"]

        regressor = Regressor(x_train_raw, nb_epoch=epochs, ls=layer_size,
                            acts=acts, dropout_rate=dropout_rate)
        regressor.fit(x_train_raw, y_train_raw, batch_size, lr=lr, print_loss=False)
        # save_regressor(regressor)

        # train test
        train_error = regressor.score(x_train_raw, y_train_raw)

        # validation test
        valid_error = regressor.score(x_valid_raw, y_valid_raw)

        # final test
        test_error = regressor.score(x_test_raw, y_test_raw)

        def print_config():
            print(f"\n========== Config {i} ==========")
            print("mixed mse:", best_result["mse"])
            print("epochs:", epochs)
            print("acts:", acts)
            print("batch_size:", batch_size)
            print("layers:", layer_size)
            print("lr:", lr)
            print("dropout_rate:", dropout_rate)
            print("\n\n")
            print("---train test---")
            print("\nRegressor error: {}\n".format(train_error))
            print("---validation test---")
            print("\nRegressor error: {}\n".format(valid_error))
            print("---final test---")
            print("\nRegressor error: {}\n".format(test_error))
            print("\n\n\n")

        print_config()
        with open("config_and_test.log", 'a') as f:
            with redirect_stdout(f):
                print_config()

        if ret_val is None:
            ret_val = {
                "details": best_result,
                "config": {
                    "epochs": epochs,
                    "acts": acts,
                    "batch_size": batch_size,
                    "layers": layer_size,
                    "lr": lr,
                    "dropout_rate": dropout_rate,
                },
                "rmse_train": train_error,
                "rmse_valid": valid_error,
                "rmse_test": test_error,
            }
    return ret_val


def multiple_train(loops=5, filename="tmp.log", valid_rmse_weight=2):
    """
    Helps analyse ray tuning result.
    Performs `loops` ray_tuning and make some recordings.
    Some results will be written to a log file.
    """
    if os.path.exists(filename):
        return
    from time import time
    start_time = time()
    results = []
    overfits_valid = []
    overfits_test = []
    rmses_train = []
    rmses_valid = []
    rmses_test = []
    for i in range(loops):
        result = ray_tuning(i+1, valid_rmse_weight)
        results.append(result)
        overfit_valid = result["rmse_valid"] / result["rmse_train"]
        overfit_test = result["rmse_test"] / result["rmse_train"]
        rmse_train = result["rmse_train"]
        rmse_valid = result["rmse_valid"]
        rmse_test = result["rmse_test"]
        overfits_valid.append(overfit_valid)
        overfits_test.append(overfit_test)
        rmses_train.append(rmse_train)
        rmses_valid.append(rmse_valid)
        rmses_test.append(rmse_test)

    end_time = time()
    duration = end_time - start_time

    def prints():
        print("Time Taken =", duration, "s\n")
        print("==================== Summary ====================")

        for result in results:
            print("config:")
            print(result["config"])
            print("rmses (train, valid, test):")
            print(result["rmse_train"])
            print(result["rmse_valid"])
            print(result["rmse_test"])
            print("\n")

        print("==================== Final Report ====================")
        print("avg overfit_valid:", sum(overfits_valid) / len(overfits_valid))
        print("avg overfit_test:", sum(overfits_test) / len(overfits_test))
        print("avg rmse_train:", sum(rmses_train) / len(rmses_train))
        print("avg rmse_valid:", sum(rmses_valid) / len(rmses_valid))
        print("avg rmse_test:", sum(rmses_test) / len(rmses_test))

    prints()
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            prints()


if __name__ == "__main__":
    import os
    os.environ["TUNE_RESULT_DIR"] = os.getcwd() + "/tmp/ray_results"
    num_loops = 1
    for i in range(1, 2):
        filename = f"loop{num_loops}_i{i}_result.log"
        multiple_train(num_loops, filename)
