import argparse
import json
import os

import keras as keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, regularizers
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from prepare_data import prepare_dataset

parser = argparse.ArgumentParser(description='Prepare Data')
parser.add_argument('-d', '--dataset', default='Cirrhosis', help='Name of dataset in data folder.')
parser.add_argument('-n', '--splits', default=10, type=int, help='Number of cross validated splits.')
parser.add_argument('-s', '--sets', default=10, type=int, help='Number of datasets to generate')
parser.add_argument('-e', '--epochs', default=400, type=int, help='Number of epochs.')
parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch Size')
args = parser.parse_args()

sets_num = args.sets
splits_num = args.splits
epochs_num = args.epochs
batch_size = args.batch_size
dataset = args.dataset

result_filepath = os.path.join(os.pardir, 'result')

def create_popphy_cnn(input_data):
    rows = input_data.shape[1]  # must be as image Height
    cols = input_data.shape[2]  # must be as image Width

    model = keras.models.Sequential()

    # ConvPoolLayer(activation_fn=ReLU, image_shape=(mini_batch_size, 1, rows, cols),
    #              filter_shape=(64, 1, 5, 10),
    #              poolsize=(2, 2)),
    model.add(Conv2D(64, (5, 10), activation='relu', input_shape=(rows, cols, 1),
                     strides=(1, 1), data_format='channels_last'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 11, 47),
    #               filter_shape=(64, 64, 4, 10),
    #               poolsize=(2, 2)),
    model.add(Conv2D(64, (4, 10), activation='relu', strides=(1, 1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 4, 19),
    #               filter_shape=(64, 64, 3, 10),
    #               poolsize=(2,2)),
    model.add(Conv2D(64, (3, 10), activation='relu', strides=(1, 1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # FullyConnectedLayer(activation_fn=ReLU, n_in=64*1*5, n_out=1024, p_dropout=0.5),
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))

    # FullyConnectedLayer( n_in=1024, n_out=1024, activation_fn=ReLU, p_dropout=0.5),
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))

    # SoftmaxLayer(n_in=1024, n_out=2, p_dropout=0.5)], mini_batch_size, c_prob)
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l2(0.1)

    # net.SGD(train, num_epochs, mini_batch_size, 0.001, validation, test, lmbda=0.1)
    sgd = SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def plain_train_metrics():
    class Metrics(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self._data = []

        def on_epoch_end(self, batch, logs={}):
            X_val, y_val = self.validation_data[0], self.validation_data[1]
            self._data.append(self.calculate(self.model, X_val, y_val))
            return

        def calculate(self, model, X_val, y_val):
            y_predict = np.asarray(model.predict(X_val))
            y_val = np.argmax(y_val, axis=1)
            y_predict = np.argmax(y_predict, axis=1)

            result = {
                'val_acc': accuracy_score(y_val, y_predict),
                'val_recall': recall_score(y_val, y_predict),
                'val_precision': precision_score(y_val, y_predict),
                'val_f1': f1_score(y_val, y_predict),
                'val_roc_auc': roc_auc_score(y_val, y_predict),
            }
            return result

        def get_data(self):
            return self._data

    return Metrics()


def train_plain(X, y):
    kfold = StratifiedKFold(n_splits=splits_num, shuffle=True)
    kfold_split = kfold.split(X, y)
    # format to Keras input shape
    X = X.reshape(X.shape + (1,))
    y = keras.utils.to_categorical(y, num_classes=2)
    best_metrics_list = []

    for index, (train_indices, val_indices) in enumerate(kfold_split):
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        model = create_popphy_cnn(X)
        best_model_filepath = os.path.join(result_filepath, 'cirrhosis' + str(index) + '.hdf5')
        metrics = plain_train_metrics()
        callback_list = [
            ModelCheckpoint(best_model_filepath, save_best_only=True, monitor='val_acc', mode='max'),
            metrics
        ]
        history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size, verbose=0,
                            validation_data=(X_val, y_val), callbacks=callback_list)

        model.load_weights(best_model_filepath)
        best_metrics_list.append(metrics.calculate(model, X_val, y_val))

        history_filepath = os.path.join(result_filepath, 'cirrhosis_hist' + str(index) + '.csv')
        with open(history_filepath, 'w') as hist_file:
            metrics_df = pd.DataFrame(metrics.get_data())
            metrics_df.drop(['val_acc'], axis=1, inplace=True) # it's duplicated in history_df
            history_df = pd.DataFrame(history.history)
            train_df = pd.concat([history_df, metrics_df], axis=1)
            train_df.to_csv(hist_file, header=True, sep=';', index=False)

    best_metrics_df = pd.DataFrame(best_metrics_list)
    best_metrics_filepath = os.path.join(result_filepath, 'best_metrics.csv')
    best_metrics_df.to_csv(best_metrics_filepath, header=True, sep=';', index=False)


def train_cross_val_score(X, y):
    kfold = StratifiedKFold(n_splits=splits_num, shuffle=True)
    kfold_split = kfold.split(X, y)
    # format to Keras input shape
    X = X.reshape(X.shape + (1,))
    y = keras.utils.to_categorical(y, num_classes=2)

    X_stratified = []
    y_stratified = []
    for index, (train_indices, val_indices) in enumerate(kfold_split):
        X_val = X[val_indices]
        y_val = y[val_indices]
        X_stratified.append(X_val)
        y_stratified.append(y_val)

    X_stratified = np.concatenate(tuple(X_stratified))
    y_stratified = np.concatenate(tuple(y_stratified))

    # cc = functools.partial(create_popphy_cnn, X_stratified)
    create_cnn = lambda: create_popphy_cnn(X_stratified)
    estimator = KerasClassifier(build_fn=create_cnn, epochs=epochs_num,
                                batch_size=batch_size, verbose=2)
    kfold = KFold(n_splits=splits_num)
    results = cross_val_score(estimator, X_stratified, y_stratified, cv=kfold)
    print results


X, y = prepare_dataset(dataset)
# train_cross_val_score(X, y)
train_plain(X, y)


# TODO use a `binary` classifier as an alternative.
# TODO simple dataset and show that both trains train_cross_val_score and train_plain work equally
