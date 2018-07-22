import os

import keras
import keras.backend as K
import numpy as np
import pandas as pd

import command_args
from graph import Graph
from prepare_data import data_path
from prepare_data import prepare_dataset
from util import create_dir

args = command_args.parse()

theta1 = 0.3  # 0.5	0.69
theta2 = 0.3  # 0\0.8

# Get reference graph
g = Graph()
g.build_graph(os.path.join(data_path, args.dataset, 'newick.txt'))


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
    for class_id in classes:
        result[class_id] = []
    for i, class_id in enumerate(y):
        result[class_id].append(data[i])
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
    g_num_nodes = g.get_node_count()
    max_indexes = np.zeros((len(classes), feature_maps_num, feature_map_flat_size))
    for klass in classes:
        for sample in feature_maps_by_class[klass]:
            for feature_i in range(0, feature_maps_num):
                maximums = np.argsort(sample[:, :, feature_i].flatten())[::-1]
                for l in range(0, int(round(theta1 * g_num_nodes))):
                    max_indexes[klass][feature_i][maximums[l]] += 1
    return max_indexes


def get_features_stats_df():
    otu_list = load_otu_list()
    d = {'OTU': otu_list, 'Max Score': np.zeros(len(otu_list)), 'Cumulative Score': np.zeros(len(otu_list))}
    return pd.DataFrame(data=d)


def get_max_active_features(feature_map_max_indexes, feature_map_width, X, y, w):
    classes = get_classes(y)
    X_by_class = split_by_class(X, y)
    df = get_features_stats_df()
    g_ref = g.get_ref()
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
                    row = loc / feature_map_width
                    col = loc % feature_map_width
                    ref_window = g_ref[row:row + w_row, col:col + w_col]
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


def calc_features_scores(max_active_features, y):
    classes = get_classes(y)
    diff = {}
    df = get_features_stats_df()

    for i in classes:
        diff[i] = df.set_index('OTU')
        for j in max_active_features[i].index:
            for k in classes:
                if i != k:
                    if j in max_active_features[k].index:
                        diff[i].loc[j, 'Max Score'] = max_active_features[i].loc[j, 'Max Score'] - \
                                                      max_active_features[k].loc[j, 'Max Score']
                        diff[i].loc[j, 'Cumulative Score'] = max_active_features[i].loc[j, 'Cumulative Score'] - \
                                                             max_active_features[k].loc[
                                                                 j, 'Cumulative Score']
                    else:
                        diff[i].loc[j, 'Max Score'] = max_active_features[i].loc[j, 'Max Score']
                        diff[i].loc[j, 'Cumulative Score'] = max_active_features[i].loc[j, 'Cumulative Score']
    return diff


def load_label_ref():
    return np.loadtxt(
        os.path.join(data_path, args.dataset, 'label_reference.txt'),
        dtype=str
    )


def write_features_scores(rankings, scores, result_dir):
    medians = {}
    class_labels = load_label_ref()
    features_dir = create_dir(result_dir, 'features')

    for class_id, _ in enumerate(class_labels):
        medians[class_id] = {}
        for j in rankings[class_id]:
            medians[class_id][j] = np.median(rankings[class_id][j])

        f = open(os.path.join(features_dir, class_labels[class_id] + '_ranklist.out'), 'w')
        for m in sorted(medians[class_id], key=medians[class_id].__getitem__):
            f.write(m + ',')
        f.close()

        f = open(os.path.join(features_dir, class_labels[class_id] + '_medians.out'), 'w')
        for m in rankings[class_id]:
            f.write(m + '\t' + str(rankings[class_id][m]) + '\n')
        f.close()

        f = open(os.path.join(features_dir, class_labels[class_id] + '_scores.out'), 'w')
        for m in scores[class_id]:
            f.write(m + '\t' + str(scores[class_id][m]) + '\n')
        f.close()


def init_feature_scores(g_node_names):
    rankings = {}
    scores = {}
    class_labels = load_label_ref()
    for class_id, _ in enumerate(class_labels):
        rankings[class_id] = {}
        scores[class_id] = {}
        for j in g_node_names:
            rankings[class_id][j] = []
            scores[class_id][j] = []
    return rankings, scores


def feature_map_stats(model_dir, i):
    model = load_model(model_dir)
    X, y, _, _ = load_data(model_dir)
    X = X.reshape(X.shape + (1,))
    # feature_maps.shape == (210, 22, 95, 64)
    feature_maps = get_feature_maps(model, X)
    feature_map_max_indexes = get_feature_map_max_indexes(feature_maps, y)
    w = model.layers[0].get_weights()[0]
    max_active_features = get_max_active_features(feature_map_max_indexes, feature_maps.shape[2], X, y, w)
    return calc_features_scores(max_active_features, y)


def main():
    result_dir = os.path.join(os.pardir, args.result_dir, args.dataset)
    g_node_names = g.get_dictionary()
    rankings, scores = init_feature_scores(g_node_names)
    for file in os.listdir(result_dir):
        if file.isdigit():
            model_dir = os.path.join(result_dir, file)
            stats = feature_map_stats(model_dir, int(file))
            for class_id in stats:
                rank = stats[class_id]['Max Score'].rank(ascending=False)
                for j in g_node_names:
                    if j in rank.index:
                        rankings[class_id][j].append(rank.loc[j])
                        scores[class_id][j].append(stats[class_id].loc[j, 'Max Score'])
                    else:
                        rankings[class_id][j].append(rank.shape[0] + 1)
                        scores[class_id][j].append(0)
    write_features_scores(rankings, scores, result_dir)


main()
