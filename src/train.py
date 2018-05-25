import argparse
import json

import keras as keras
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, regularizers
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
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


def train_plain(X, y):
    kfold = StratifiedKFold(n_splits=splits_num, shuffle=True)
    kfold_split = kfold.split(X, y)
    # format to Keras input shape
    X = X.reshape(X.shape + (1,))
    y = keras.utils.to_categorical(y, num_classes=2)
    val_acc_list = []
    val_loss_list = []

    for index, (train_indices, val_indices) in enumerate(kfold_split):
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        model = create_popphy_cnn(X)
        best_model_filepath = os.path.join(os.pardir, 'result', 'cirrhosis' + str(index) + '.hdf5')
        callback_list = [
            ModelCheckpoint(best_model_filepath, save_best_only=True, monitor='val_acc', mode='max')
        ]
        history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size, verbose=0,
                            validation_data=(X_val, y_val), callbacks=callback_list)

        model.load_weights(best_model_filepath)
        score = model.evaluate(X_val, y_val)
        val_loss_list.append(score[0])
        val_acc_list.append(score[1])

        history_filepath = os.path.join(os.pardir, 'result', 'cirrhosis_hist' + str(index) + '.txt')
        with open(history_filepath, 'w') as hist_file:
            json.dump(history.history, hist_file)

    val_acc_list = np.array(val_acc_list)
    val_loss_list = np.array(val_loss_list)
    print '\n######################################################################'
    print 'val_acc_list mean and std: ', val_acc_list.mean(), val_acc_list.std()
    print 'val_loss_list mean and std: ', val_loss_list.mean(), val_loss_list.std()


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
