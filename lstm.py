from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from load_dataset import load_data

DATASET_PATH = r"C:\Users\Admin\Documents\An_4\Semestrul_2\Licenta\datasets\data_extended.json"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prepare_datasets(test_size, validation_size):
    # load data
    X, Y = load_data(DATASET_PATH)

    # create the train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


def build_model(input_shape):
    """
    Create RNN (LSTM) model
    :param input_shape: 130 x 13 dimension
    """
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # feed into dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer (with softmax)
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(0.1, 0.1)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # compile the net
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # train the net
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=32, epochs=100)

    model.save('lstm.h5')

    # evaluate the net on the test set
    test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))