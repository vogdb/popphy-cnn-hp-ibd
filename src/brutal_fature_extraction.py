# this is an adapted copy of original `feature_map_analysis`
# delete it as soon as the simplified version is ready
import argparse
import os

import keras
import keras.backend as K
import numpy as np
import pandas as pd

from graph import Graph
from prepare_data import data_path
from prepare_data import prepare_dataset


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


def get_feature_maps(model, input):
    get_layer0_output = K.function([model.layers[0].input], [model.layers[0].output])
    return get_layer0_output([input])[0]


parser = argparse.ArgumentParser(description="PopPhy-CNN Feature Extraction")
parser.add_argument("-d", "--dataset", default="Cirrhosis", help="Name of dataset in data folder.")
args = parser.parse_args()

dset = args.dataset
num_splits = 10
num_sets = 1
dataset_path = os.path.join(data_path, args.dataset)

# Get reference graph
g = Graph()
g.build_graph(os.path.join(dataset_path, 'newick.txt'))
ref = g.get_ref()
num_nodes = g.get_node_count()

rankings = {}
scores = {}
node_names = g.get_dictionary()

labels = np.loadtxt(
    os.path.join(dataset_path, 'label_reference.txt'),
    dtype=str
)
otus = np.loadtxt(
    os.path.join(dataset_path, 'otu.csv'),
    dtype=str,
    delimiter='\n',
)
result_dir = os.path.join(os.pardir, 'result', '400e_model_nofc', args.dataset)
num_classes = len(labels)

for i in range(0, len(labels)):
    rankings[i] = {}
    scores[i] = {}
    for j in node_names:
        rankings[i][j] = []
        scores[i][j] = []

for roc in range(0, num_sets * num_splits):
    set = str(roc / num_splits)
    cv = str(roc % num_splits)
    model_dir = os.path.join(result_dir, str(roc))
    net = load_model(model_dir)
    X_train, y_train, _, _ = load_data(model_dir)
    X_train = X_train.reshape(X_train.shape + (1,))

    num_train = X_train.shape[0]
    # num_test = test[1].eval().shape[0]
    num_samp = num_train
    w = net.layers[0].get_weights()[0]
    num_maps = w.shape[3]
    w_row = w.shape[0]
    w_col = w.shape[1]

    data_set = X_train
    data_shape = data_set[0].shape

    fm = {}
    data = {}
    for i in range(0, num_classes):
        fm[i] = []
        data[i] = []

    f_list = get_feature_maps(net, data_set)
    for i in range(0, num_samp):
        f = f_list[i]
        y = y_train[i]
        fm[int(y)].append(f)
        data[int(y)].append(data_set[i])

    for i in range(0, num_classes):
        fm[i] = np.array(fm[i])
        data[i] = np.array(data[i])

    fm_rows = fm[0][0].shape[0]
    fm_cols = fm[0][0].shape[1]

    theta1 = 0.3  # 0.5	0.69
    theta2 = 0.3  # 0\0.8

    # Get the top X max indices for each class and each feature map
    max_list = np.zeros((num_classes, num_maps, fm_rows * fm_cols))

    for i in range(0, num_classes):
        for j in range(0, len(fm[i])):
            for k in range(0, num_maps):
                maximums = np.argsort(fm[i][j][:, :, k].flatten())[::-1]
                for l in range(0, int(round(theta1 * num_nodes))):
                    max_list[i][k][maximums[l]] += 1

    # np.savetxt('brutal_feature_extraction/max_list_0_{}{}.txt'.format(set, cv), max_list[0], fmt='%u')
    # np.savetxt('brutal_feature_extraction/max_list_1_{}{}.txt'.format(set, cv), max_list[1], fmt='%u')
    d = {"OTU": otus, "Max Score": np.zeros(len(otus)), "Cumulative Score": np.zeros(len(otus))}
    df = pd.DataFrame(data=d)
    results = {}

    for i in range(0, num_classes):
        results[i] = df.set_index("OTU")
    # For each class
    for i in range(0, num_classes):

        # For each feature map...
        for j in range(0, num_maps):

            # Order the feature map's maximums
            loc_list = max_list[i][j].argsort()[::-1]

            # Store kernel weights
            w = net.layers[0].get_weights()[0][:, :, 0, j]
            # w = np.rot90(np.rot90(net.layers[0].w.container.data[0][0]))

            # For the top X maximums...
            for k in range(0, len(loc_list)):

                # Find the row and column location and isolate reference window
                loc = loc_list[k]
                if max_list[i][j][loc] > int(round(len(fm[i]) * theta2)):
                    row = loc / fm_cols
                    col = loc % fm_cols
                    ref_window = ref[row:row + w_row, col:col + w_col]
                    count = np.zeros((w_row, w_col))

                    # Calculate the proportion of the contribution of each pixel to the convolution with the absolute value of weights
                    for l in range(0, len(fm[i])):
                        window = np.squeeze(data[i][l][row:row + w_row, col:col + w_col])
                        abs_v = (abs(w) * window).sum()
                        v = (w * window)
                        for m in range(0, v.shape[0]):
                            for n in range(0, v.shape[1]):
                                count[m, n] += v[m, n] / abs_v

                    # Divide by number of samples
                    count = count / len(fm[i])

                    # Print out features with a high enough value
                    for m in range(0, w_row):
                        for n in range(0, w_col):
                            if count[m, n] > 0:
                                if ref_window[m, n] in results[i].index:
                                    if count[m, n] > results[i].loc[ref_window[m, n], "Max Score"]:
                                        results[i].loc[ref_window[m, n], "Max Score"] = count[m, n]
                                    results[i].loc[ref_window[m, n], "Cumulative Score"] += count[m, n]
                                else:
                                    results[i].loc[ref_window[m, n], "Max Score"] = count[m, n]
                                    results[i].loc[ref_window[m, n], "Cumulative Score"] = count[m, n]

    results[0].to_csv('brutal_feature_extraction/results_0_{}{}'.format(set, cv), header=True, sep=';', index=True)
    results[1].to_csv('brutal_feature_extraction/results_1_{}{}'.format(set, cv), header=True, sep=';', index=True)
    diff = {}

    for i in range(0, num_classes):
        diff[i] = df.set_index("OTU")
        for j in results[i].index:
            for k in range(0, num_classes):
                if i != k:
                    if j in results[k].index:
                        diff[i].loc[j, "Max Score"] = results[i].loc[j, "Max Score"] - results[k].loc[j, "Max Score"]
                        diff[i].loc[j, "Cumulative Score"] = results[i].loc[j, "Cumulative Score"] - results[k].loc[
                            j, "Cumulative Score"]
                    else:
                        diff[i].loc[j, "Max Score"] = results[i].loc[j, "Max Score"]
                        diff[i].loc[j, "Cumulative Score"] = results[i].loc[j, "Cumulative Score"]

        rank = diff[i]["Max Score"].rank(ascending=False)
        for j in node_names:
            if j in rank.index:
                rankings[i][j].append(rank.loc[j])
                scores[i][j].append(diff[i].loc[j, "Max Score"])
            else:
                rankings[i][j].append(rank.shape[0] + 1)
                scores[i][j].append(0)

medians = {}

for i in range(0, num_classes):
    medians[i] = {}
    for j in rankings[i]:
        medians[i][j] = np.median(rankings[i][j])

    f = open("brutal_feature_extraction/" + labels[i] + "_ranklist.out", "w")
    for m in sorted(medians[i], key=medians[i].__getitem__):
        f.write(m + ",")
    f.close()

    f = open("brutal_feature_extraction/" + labels[i] + "_medians.out", "w")
    for m in rankings[i]:
        f.write(m + "\t" + str(rankings[i][m]) + "\n")
    f.close()

    f = open("brutal_feature_extraction/" + labels[i] + "_scores.out", "w")
    for m in scores[i]:
        f.write(m + "\t" + str(scores[i][m]) + "\n")
    f.close()

print("Finished")
