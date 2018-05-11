import argparse

from prepare_data import prepare_dataset

parser = argparse.ArgumentParser(description='Prepare Data')
parser.add_argument('-d', '--dataset', default='Cirrhosis', help='Name of dataset in data folder.')
parser.add_argument('-n', '--splits', default=10, type=int, help='Number of cross validated splits.')
parser.add_argument('-s', '--sets', default=10, type=int, help='Number of datasets to generate')
args = parser.parse_args()

sets_num = args.sets
splits_num = args.splits
dataset = args.dataset
X, y = prepare_dataset(dataset)
