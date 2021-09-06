import tensorflow as tf
import tensorflow.keras as keras
from load_dataset import prepare_datasets

DATASET_PATH = r"C:\Users\Admin\Documents\An_4\Semestrul_2\Licenta\datasets\data_extended.json"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_model(input_shape):

    # Input
    inputs = keras.layers.Input(shape=input_shape)

    # CNN Block
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    pool11 = keras.layers.MaxPool2D((3,3), strides=(2,2))(conv1)
    batch1 = keras.layers.BatchNormalization()(pool11)

    conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(batch1)
    pool12 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(conv2)
    batch2 = keras.layers.BatchNormalization()(pool12)

    conv3 = keras.layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu')(batch2)
    pool13 = keras.layers.MaxPool2D((2, 2), strides=(2, 2))(conv3)
    batch3 = keras.layers.BatchNormalization()(pool13)

    flatten = keras.layers.Flatten()(batch3)

    # Remove channel axis so we can pass the data into RNN
    squeezed = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, axis=-1))(inputs)

    # RNN - LSTM
    lstm1 = keras.layers.LSTM(64, return_sequences=True)(squeezed)
    lstm2 = keras.layers.LSTM(64)(lstm1)

    # Concat Output
    concat = keras.layers.concatenate([flatten, lstm2])

    # Softmax Output
    output = keras.layers.Dense(10, activation='softmax')(concat)

    model = keras.Model(inputs=inputs, outputs=[output])
    return model


if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(0.1, 0.1)

    # build the net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the net
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # train the net
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=16, epochs=100)

    model.save('lstm_cnn.h5')

    # evaluate the net on the test set
    test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))