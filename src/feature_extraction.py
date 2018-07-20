import os

import keras
import keras.backend as K
import numpy as np
import pandas as pd

import command_args
from graph import Graph
from prepare_data import data_path
from prepare_data import prepare_dataset

args = command_args.parse()

theta1 = 0.3  # 0.5	0.69
theta2 = 0.3  # 0\0.8

# Get reference graph
g = Graph()
g.build_graph(os.path.join(data_path, args.dataset, 'newick.txt'))
ref = g.get_ref()
num_nodes = g.get_node_count()


def get_feature_maps(model, input):
    get_layer0_output = K.function([model.layers[0].input], [model.layers[0].output])
    return get_layer0_output([input])[0]


def load_model(model_dir):
    model_file = os.path.join(model_dir, 'network.hdf5')
    return keras.models.load_model(model_file)


def load_data(model_dir):
    indices_file = os.path.join(model_dir, 'data_indices.npz')
    indices = np.load(indices_file)
    train_indices = indices['train']
    val_indices = indices['val']
    X, y = prepare_dataset(args.dataset)
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def load_otu_list():
    return np.loadtxt(
        os.path.join(os.path.join(data_path, args.dataset), 'otu.csv'),
        dtype=str,
        delimiter='\n',
    )


def get_classes(y):
    return sorted(set(y))


def split_by_class(data, y):
    classes = get_classes(y)
    result = {}
    for klass in classes:
        result[klass] = []
    for i, y_i in enumerate(y):
        result[y_i].append(data[i])
    return result


def get_feature_map_max_indexes(feature_maps, y):
    '''
    Calculates count of max values indexes for each feature map.
    :param feature_maps:
    :param y:
    :return:
        {'class':
            {feature_map:
                index of max value where each index is represented by its counter, e.g. we count how much the index has been encountered
            }
        }
    '''
    classes = get_classes(y)
    feature_maps_by_class = split_by_class(feature_maps, y)
    feature_maps_num = feature_maps.shape[3]
    feature_map_flat_size = feature_maps.shape[1] * feature_maps.shape[2]

    # count the top `l` max indices for each class and each feature map
    max_indexes = np.zeros((len(classes), feature_maps_num, feature_map_flat_size))
    for klass in classes:
        for sample in feature_maps_by_class[klass]:
            for feature_i in range(0, feature_maps_num):
                maximums = np.argsort(sample[:, :, feature_i].flatten())[::-1]
                for l in range(0, int(round(theta1 * num_nodes))):
                    max_indexes[klass][feature_i][maximums[l]] += 1
    return max_indexes


def get_max_active_features(feature_map_max_indexes, feature_maps, X, y, w):
    classes = get_classes(y)
    X_by_class = split_by_class(X, y)

    otu_list = load_otu_list()
    d = {'OTU': otu_list, 'Max Score': np.zeros(len(otu_list)), 'Cumulative Score': np.zeros(len(otu_list))}
    df = pd.DataFrame(data=d)
    results = {}

    for i in classes:
        results[i] = df.set_index('OTU')

    # For each class
    for i in classes:
        # For each feature map...
        for j in range(0, len(feature_map_max_indexes[i])):
            # Get indexes that have been encountered the most.
            loc_list = feature_map_max_indexes[i][j].argsort()[::-1]

            # kernel weights of `j` feature map
            w_j = w[:, :, 0, j]
            w_row = w_j.shape[0]
            w_col = w_j.shape[1]

            # For the top maximums
            for k in range(0, len(loc_list)):

                # Find the row and column location and isolate reference window
                loc = loc_list[k]
                if feature_map_max_indexes[i][j][loc] > int(round(len(X_by_class[i]) * theta2)):
                    row = loc / feature_maps.shape[2]
                    col = loc % feature_maps.shape[2]
                    ref_window = ref[row:row + w_row, col:col + w_col]
                    count = np.zeros((w_row, w_col))

                    # Calculate the proportion of the contribution of each input pixel to the convolution with the absolute value of weights
                    for l in range(0, len(X_by_class[i])):
                        window = np.squeeze(X_by_class[i][l][row:row + w_row, col:col + w_col])
                        abs_v = (abs(w_j) * window).sum()
                        v = (w_j * window)
                        for m in range(0, v.shape[0]):
                            for n in range(0, v.shape[1]):
                                count[m, n] += v[m, n] / abs_v

                    # Divide by number of samples
                    count = count / len(X_by_class[i])

                    # Print out features with a high enough value
                    for m in range(0, w_row):
                        for n in range(0, w_col):
                            if count[m, n] > 0:
                                if ref_window[m, n] in results[i].index:
                                    if count[m, n] > results[i].loc[ref_window[m, n], 'Max Score']:
                                        results[i].loc[ref_window[m, n], 'Max Score'] = count[m, n]
                                    results[i].loc[ref_window[m, n], 'Cumulative Score'] += count[m, n]
                                else:
                                    results[i].loc[ref_window[m, n], 'Max Score'] = count[m, n]
                                    results[i].loc[ref_window[m, n], 'Cumulative Score'] = count[m, n]
    return results


def feature_map_stats(model_dir, i):
    model = load_model(model_dir)
    X, y, _, _ = load_data(model_dir)
    X = X.reshape(X.shape + (1,))
    # feature_maps.shape == (210, 22, 95, 64)
    feature_maps = get_feature_maps(model, X)
    feature_map_max_indexes = get_feature_map_max_indexes(feature_maps, y)
    w = model.layers[0].get_weights()[0]
    max_active_features = get_max_active_features(feature_map_max_indexes, feature_maps, X, y, w)
    # max_active_features[0].to_csv('feature_extraction/results_0_0{}'.format(i), header=True, sep=';', index=True)
    # max_active_features[1].to_csv('feature_extraction/results_1_0{}'.format(i), header=True, sep=';', index=True)

    raise SystemExit(3)


result_dir = os.path.join(os.pardir, 'result', '400e_model_nofc', args.dataset)
file = '0'
# for file in os.listdir(result_dir):
#     if file.isdigit():
model_dir = os.path.join(result_dir, file)  # TODO file instead of '0'
feature_map_stats(model_dir, int(file))
