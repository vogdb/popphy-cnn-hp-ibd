import os

import keras as keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from prepare_data import prepare_dataset
import create_model
import command_args

args = command_args.parse()
result_filepath = os.path.join(os.pardir, 'result')


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


def train_plain(X, y):
    kfold = StratifiedKFold(n_splits=args.splits, shuffle=True)
    kfold_split = kfold.split(X, y)
    # format to Keras input shape
    X = X.reshape(X.shape + (1,))
    # y = keras.utils.to_categorical(y, num_classes=2)
    best_metrics_list = []

    for index, (train_indices, val_indices) in enumerate(kfold_split):
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        model = create_model.binary_model(X)
        best_model_filepath = os.path.join(result_filepath, 'cirrhosis' + str(index) + '.hdf5')
        metrics = plain_train_metrics()
        callback_list = [
            ModelCheckpoint(best_model_filepath, save_best_only=True, monitor='val_acc', mode='max'),
            metrics
        ]
        history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0,
                            validation_data=(X_val, y_val), callbacks=callback_list)

        model.load_weights(best_model_filepath)
        best_metrics_list.append(metrics.calculate(model, X_val, y_val))

        history_filepath = os.path.join(result_filepath, 'cirrhosis_hist' + str(index) + '.csv')
        with open(history_filepath, 'w') as hist_file:
            metrics_df = pd.DataFrame(metrics.get_data())
            metrics_df.drop(['val_acc'], axis=1, inplace=True)  # it's duplicated in history_df
            history_df = pd.DataFrame(history.history)
            train_df = pd.concat([history_df, metrics_df], axis=1)
            train_df.to_csv(hist_file, header=True, sep=';', index=False)

    best_metrics_df = pd.DataFrame(best_metrics_list)
    best_metrics_filepath = os.path.join(result_filepath, 'best_metrics.csv')
    best_metrics_df.to_csv(best_metrics_filepath, header=True, sep=';', index=False)


def train_cross_val_score(X, y):
    kfold = StratifiedKFold(n_splits=args.splits, shuffle=True)
    kfold_split = kfold.split(X, y)
    # format to Keras input shape
    X = X.reshape(X.shape + (1,))
    # y = keras.utils.to_categorical(y, num_classes=2)

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
    create_cnn = lambda: create_model.onehot_model(X_stratified)
    estimator = KerasClassifier(build_fn=create_cnn, epochs=args.epochs,
                                batch_size=args.batch_size, verbose=2)
    kfold = KFold(n_splits=args.splits)
    results = cross_val_score(estimator, X_stratified, y_stratified, cv=kfold)
    print results


X, y = prepare_dataset(args.dataset)
# train_cross_val_score(X, y)
train_plain(X, y)
