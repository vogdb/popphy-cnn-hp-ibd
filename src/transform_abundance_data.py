import numpy as np
import os

data_path = '../data'


def transform(dataset_name):
    dataset_path = os.path.join(data_path, dataset_name)
    abundance_path = os.path.join(dataset_path, 'abundance.txt')
    abundance_dict = {}

    for line in open(abundance_path):
        split = line.split('\t')

        if 'cirrhosis' in dataset_name.lower():
            otu = extract_otu_cirrhosis(split)

        count_raw = np.array(split[1:], dtype=float)
        if otu in abundance_dict:
            abundance_dict[otu] = np.add(abundance_dict[otu], count_raw)
        else:
            abundance_dict[otu] = count_raw

    otu_list = abundance_dict.keys()
    count_matrix = np.vstack(abundance_dict.values()).T

    write_otu(dataset_path, otu_list)
    write_count_matrix(dataset_path, count_matrix)


def write_otu(dataset_path, otu_list):
    otu_path = os.path.join(dataset_path, 'otu.csv')
    with open(otu_path, 'w') as otu_file:
        otu_file.write(','.join(map(str, otu_list)))


def write_count_matrix(dataset_path, count_matrix):
    matrix_path = os.path.join(dataset_path, 'count_matrix.csv')
    with open(matrix_path, 'w') as matrix_file:
        for i in range(0, len(count_matrix)):
            matrix_file.write(','.join(map(str, count_matrix[i])))
            matrix_file.write('\n')


def extract_otu_cirrhosis(split):
    otu = split[0].split('g__')[1]
    otu = otu.split('_noname')[0].replace('_', ' ').replace('unclassified', '').replace(
        'Clostridiales Family XIII Incertae Sedis', 'Clostridiales Family XIII. Incertae Sedis').strip()
    return otu


# test usage example
if __name__ == "__main__":
    transform('Cirrhosis')
