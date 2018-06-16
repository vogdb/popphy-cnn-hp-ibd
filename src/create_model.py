import keras as keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, regularizers
from keras.optimizers import SGD


def shared_model_pre(input_data):
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
    return model


def regularize_model(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l2(0.1)


def binary_model(input_data):
    model = shared_model_pre(input_data)
    model.add(Dense(1, activation='sigmoid'))
    regularize_model(model)

    sgd = SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def onehot_model(input_data):
    model = shared_model_pre(input_data)
    model.add(Dense(2, activation='softmax'))
    regularize_model(model)

    sgd = SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
