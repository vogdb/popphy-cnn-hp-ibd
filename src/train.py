import os

import keras as keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from prepare_data import prepare_dataset
import create_model
import command_args

args = command_args.parse()
result_dir = os.path.join(os.pardir, 'result')


def train_metrics():
    class Metrics(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self._data = []

        def on_epoch_end(self, batch, logs={}):
            X_val, y_val = self.validation_data[0], self.validation_data[1]
            self._data.append(self.calculate(self.model, X_val, y_val))
            return

        def calculate(self, model, X_val, y_val):
            y_predict = np.asarray(model.predict(X_val))
            y_val = np.squeeze(y_val)
            y_predict = [int(round(y_i)) for y_i in np.squeeze(y_predict)]

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


def create_model_dir(result_dir, index):
    model_dir = os.path.join(result_dir, args.dataset, str(index))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def save_data_indices(model_dir, train_indices, val_indices):
    indeces_filepath = os.path.join(model_dir, 'data_indeces')
    np.savez(indeces_filepath, train=train_indices, val=val_indices)


def train(X, y):
    kfold = StratifiedKFold(n_splits=args.splits, shuffle=True)
    kfold_split = kfold.split(X, y)
    # format to Keras input shape
    X = X.reshape(X.shape + (1,))
    # y = keras.utils.to_categorical(y, num_classes=2)
    best_metrics_list = []

    for index, (train_indices, val_indices) in enumerate(kfold_split):
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        model_dir = create_model_dir(result_dir, index)
        save_data_indices(model_dir, train_indices, val_indices)
        model = create_model.binary_model(X)
        best_model_filepath = os.path.join(model_dir, 'network.hdf5')
        metrics = train_metrics()
        callback_list = [
            ModelCheckpoint(best_model_filepath, save_best_only=True, monitor='val_acc', mode='max'),
            metrics
        ]
        history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0,
                            validation_data=(X_val, y_val), callbacks=callback_list)

        model.load_weights(best_model_filepath)
        best_metrics_list.append(metrics.calculate(model, X_val, y_val))

        history_filepath = os.path.join(model_dir, 'history.csv')
        with open(history_filepath, 'w') as hist_file:
            metrics_df = pd.DataFrame(metrics.get_data())
            metrics_df.drop(['val_acc'], axis=1, inplace=True)  # it's duplicated in history_df
            history_df = pd.DataFrame(history.history)
            train_df = pd.concat([history_df, metrics_df], axis=1)
            train_df.to_csv(hist_file, header=True, sep=';', index=False)

    best_metrics_df = pd.DataFrame(best_metrics_list)
    best_metrics_filepath = os.path.join(result_dir, 'best_metrics.csv')
    best_metrics_df.to_csv(best_metrics_filepath, header=True, sep=';', index=False)


X, y = prepare_dataset(args.dataset)
train(X, y)
