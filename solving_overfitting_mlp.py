import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow.keras as keras
from my_prediction import predict_whole_test_dataset
from my_plots import plot_history, plot_confusion
from load_dataset import load_data

DATASET_PATH = "data_extended.json"

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prepare_datasets(test_size, validation_size):
    # load data
    X, Y = load_data(DATASET_PATH)

    # create the train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    print(len(X_train))
    print(len(X_test))

    # create train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train,
                                                                    test_size=validation_size)

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


if __name__ == "__main__":
    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split data into train/validation/test
    inputs_train, inputs_validation, inputs_test, \
    targets_train, targets_validation, targets_test = prepare_datasets(0.1, 0.1)

    # build the network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    start = time.perf_counter()

    # train network
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_validation, targets_validation),
                        epochs=100, batch_size=32)

    training_time = time.perf_counter() - start
    print(f"Total training time: {training_time}s")

    model.save('mlp.h5')

    # evaluate the network on the test set
    test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make prediction on whole test set
    y_true_list, y_pred_list = predict_whole_test_dataset(model, inputs_test, targets_test)

    class_names = ['blues', 'classical', 'country', 'disco',
                   'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    print(classification_report(y_true_list, y_pred_list, target_names=class_names))

    #plot confusion matrix
    plot_confusion(y_true_list, y_pred_list)

    #plot error and accuracy
    plot_history(history)