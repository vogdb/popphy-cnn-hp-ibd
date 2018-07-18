import os
import os.path as path

import numpy as np
import pandas as pd
from graph import Graph

data_path = path.abspath(path.join('.', os.pardir, 'data'))


def extract_abundance_data(dataset_path):
    abundance_path = path.join(dataset_path, 'abundance.txt')
    abundance_dict = {}

    extract_otu = None
    if dataset_path.lower().endswith('cirrhosis'):
        extract_otu = extract_otu_cirrhosis

    if extract_otu is None:
        raise ValueError('Unknown dataset ' + dataset_path)

    for line in open(abundance_path):
        split = line.split('\t')
        otu = extract_otu(split)
        count_raw = np.array(split[1:], dtype=float)
        if otu in abundance_dict:
            abundance_dict[otu] = np.add(abundance_dict[otu], count_raw)
        else:
            abundance_dict[otu] = count_raw

    otu_list = abundance_dict.keys()
    count_matrix = np.vstack(abundance_dict.values()).T

    return count_matrix.astype(np.float32), otu_list


def write_otu(dataset_path, otu_list):
    otu_path = path.join(dataset_path, 'otu.csv')
    np.savetxt(otu_path, otu_list, fmt='%s')


def write_count_matrix(dataset_path, count_matrix):
    matrix_path = path.join(dataset_path, 'count_matrix.csv')
    np.savetxt(matrix_path, count_matrix, fmt='%.5f')


def write_label_dict(dataset_path, y_dict):
    label_ref_path = path.join(dataset_path, 'label_reference.txt')
    np.savetxt(label_ref_path, y_dict, fmt='%s')


def extract_otu_cirrhosis(split):
    otu = split[0].split('g__')[1]
    otu = otu.split('_noname')[0].replace('_', ' ').replace('unclassified', '').replace(
        'Clostridiales Family XIII Incertae Sedis', 'Clostridiales Family XIII. Incertae Sedis').strip()
    return otu


def convert_into_tree_matrix(X, g, f):
    result = []
    for x in X:
        g.populate_graph(f, x)
        result.append(np.array(g.get_map()))
    return np.array(result)


def prepare_dataset(dataset_name):
    """
    Args:
        dataset_name (str): dataset name as in `data` folder

    return:
        np.array, np.array: the first is the dataset in CNN-acceptable format, the second is the list of labels of the dataset.
    """

    dataset_path = path.join(data_path, dataset_name)
    X, features = extract_abundance_data(dataset_path)
    y = np.genfromtxt(path.join(dataset_path, 'labels.txt'), dtype=np.str_, delimiter=',')
    y = pd.factorize(y)

    # write some log information
    write_otu(dataset_path, features)
    write_count_matrix(dataset_path, X)
    write_label_dict(dataset_path, y[1])

    tree_matrix_path = path.join(dataset_path, 'tree_matrix.npy')
    if path.exists(tree_matrix_path):
        X_tree_matrix = np.load(tree_matrix_path)
    else:
        # Build phylogenetic tree graph
        g = Graph()
        g.build_graph(path.join(dataset_path, 'newick.txt'))
        # Convert abundance matrix into tree matrix
        X_tree_matrix = convert_into_tree_matrix(X, g, features)
        np.save(tree_matrix_path, X_tree_matrix)
    return X_tree_matrix, y[0]


# test usage example
if __name__ == "__main__":
    prepare_dataset('Cirrhosis')
