import argparse
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, regularizers
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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

X, y = prepare_dataset(dataset)


def popphy_cnn():
    # see https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    rows = X.shape[0]  # must be as image Height
    cols = X.shape[1]  # must be as image Width

    model = Sequential()

    # ConvPoolLayer(activation_fn=ReLU, image_shape=(mini_batch_size, 1, rows, cols),
    #              filter_shape=(64, 1, 5, 10),
    #              poolsize=(2, 2)),
    model.add(Conv2D(64, (10, 5), activation='relu', input_shape=(rows, cols, 1),
                     strides=(1, 1), data_format='channels_last'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 11, 47),
    #               filter_shape=(64, 64, 4, 10),
    #               poolsize=(2, 2)),
    model.add(Conv2D(64, (10, 4), activation='relu',
                     strides=(1, 1), data_format='channels_last'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 4, 19),
    #               filter_shape=(64, 64, 3, 10),
    #               poolsize=(2,2)),
    model.add(Conv2D(64, (10, 3), activation='relu',
                     strides=(1, 1), data_format='channels_last'))
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

    # net.SGD(train, num_epochs, mini_batch_size, 0.001, validation, test, lmbda=0.1)
    sgd = SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l2(0.1)

    return model


estimator = KerasClassifier(build_fn=popphy_cnn, epochs=epochs_num, batch_size=batch_size, verbose=0)
kfold = KFold(n_splits=splits_num)
results = cross_val_score(estimator, X, y, cv=kfold)

print("Results: %.2f (%.2f)" % (results.mean(), results.std()))
