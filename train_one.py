from part2_house_value_regression import save_regressor, Regressor
from utils import parse_file


def main():
    x_train, y_train = parse_file(p_train=1)
    regressor = Regressor(
        x_train,
        nb_epoch=100,
        ls=[92, 106],
        acts=['relu', 'relu'],
        dropout_rate=0.11500773415805182,
    )
    regressor.fit(
        x_train,
        y_train,
        batch_size=64,
        lr=0.0010667679657819984,
    )
    save_regressor(regressor)


if __name__ == "__main__":
    main()
