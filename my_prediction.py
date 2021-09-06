import numpy as np


def predict(model, X, Y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)

    return Y, predicted_index.tolist()


def predict_whole_test_dataset(model, X_test, Y_test):
    y_true_list = []
    y_pred_list = []

    for i in range(len(X_test)):
        y_true, y_pred = predict(model, X_test[i], Y_test[i])
        y_true_list.append(y_true)
        y_pred_list.append(y_pred[0])

    y_true_list = np.array(y_true_list)
    y_pred_list = np.array(y_pred_list)

    return y_true_list, y_pred_list