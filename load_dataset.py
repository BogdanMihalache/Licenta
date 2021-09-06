import numpy as np
import json
from sklearn.model_selection import train_test_split

DATASET_PATH = "data_extended.json"

def load_data(dataset_path):

    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def prepare_datasets(test_size, validation_size):

    # load data
    X, Y = load_data(DATASET_PATH)

    # create the train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test