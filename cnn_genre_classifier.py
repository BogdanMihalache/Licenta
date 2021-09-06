import tensorflow as tf
import tensorflow.keras as keras
from load_dataset import prepare_datasets

DATASET_PATH = "data_extended.json"

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_model(input_shape):
    """
    Create model
    :param input_shape: 130 x 13 x 1 dimension
    """
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer (with softmax)
    model.add(keras.layers.Dense(10, activation='softmax')) # creates a probability distribution for these 10 categories

    return model


if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(0.1, 0.1)

    # build the CNN
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the net
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # train the CNN
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=32, epochs=3)

    model.save('cnn.h5')

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print("Accuracy on test set is: {}". format(test_accuracy))