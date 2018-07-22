import argparse


def parse():
    parser = argparse.ArgumentParser(description='train params')
    parser.add_argument('-d', '--dataset', default='Cirrhosis', help='Name of dataset in data folder.')
    parser.add_argument('-n', '--splits', default=10, type=int, help='Number of cross validated splits.')
    parser.add_argument('-e', '--epochs', default=400, type=int, help='Number of epochs.')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('-r', '--result_dir', default='result',
                        help='Result directory path. It is used only for feature extraction.')
    return parser.parse_args()
