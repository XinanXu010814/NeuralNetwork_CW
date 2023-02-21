# Artificial Neural Network Coursework2

```sh
.
├── final_result.txt                 # the result report for part2 tuning
├── part1_nn_lib.py                  # part1 code
├── part2_house_value_regression.py  # part2 code
├── part2_model.pickle               # tuned part2 model
├── requirements.txt                 # python module requirements
├── run_tuning.sh                    # a shell script to run tuning
├── train_one.py                     # used to save the regressor with certain 
│                                    # hyperparameters to the pickle file
├── tuning.py                        # used to call the tuning function in
│                                    # "part2_house_value_regression.py"
└── utils.py                         # contains some utility functions
```

## Running Our Code

### To Run `part1_nn_lib.py` or `part2_house_value_regression.py`

If you are running it on a lab machine, execute the following commands in shell:

```sh
source /vol/lab/intro2ml/venv/bin/activate
# then either
python part1_nn_lib.py
# or
python part2_house_value_regression.py
```

Otherwise create a python venv, activate it, install dependencies in "requirements.txt" and then run python.

Or you may run: `pip install -r requirements.txt`, and then run the python files.

### To Run `tuning.py`

Make sure you have installed the dependencies in requirements.py, and then run:

```sh
python tuning.py  # this takes around 2 hours
```

Running this generates several log files:

- `report.log` gives more details on the tuning
- `config_and_test.log` gives some test metrics of the tuned result

Both files contain the best three configurations of the hyperparameter tuning.
