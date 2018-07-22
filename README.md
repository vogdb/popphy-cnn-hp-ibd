# About

This is an application of [PopPhy-CNN](https://github.com/derekreiman/PopPhy-CNN) for HP (*Helicobacter pylori*) and IBD (inflammatory bowel disease) data.

## Requirements

- Python 2.7.14
- Libraries: `pip install theano numpy pandas joblib xmltodict untangle sklearn network`

## Input data
Input data must be represented as a dataset. Datasets are stored in respective folders under the data directory. Each dataset needs the following:
  - `abundance.txt`. A space separated file representing the abundance table. Each row should represent the abundance of an OTU and each column should represent a sample.
  - `labels.txt`. Class labels of samples that ordered in the same way as in `abundance.txt`. There should be one label per line.            
  - `newick.txt`. It is the newick formatted text file for the phylogenetic taxonomic tree.             

## Training

To train the network please use `src/train.py`. You can run it with several arguments. All of them are used in the command below:
```bash
cd src
# python train.py -d=<dataset name> -n=<number of cross validation splits> -e=<number of epochs> -b=<batch size>
python train.py -d=Cirrhosis -n=10 -e=400 -b=1
```
For details please look at the `src/command_args.py`. Results of the network training are saved in the directory `result` with the name of its dataset (`result/Dataset_name`). This directory contains result directories per each cross validation split (hereinafter CV split for short). If you'd had 3 splits then it would've had 3 directories with names `0`, `1`, `2`. Each CV split directory contains files: `network.hdf5`, `history.csv`, `data_indices.npz`.

  - `network.hdf5` contains the best network state
  - `history.csv` contains all training metrics
  - `data_indices.npz` contains indices of the CV data
  
## Feature extraction

After a successful network training you can extract the best features of the dataset by running a script `src/feature_extraction.py`.
```bash
# python feature_extraction.py -d=<dataset name> -r=<result directory; optional; `result` by default>
python feature_extraction.py -d=Cirrhosis
```
The extracted features will be saved in `<result dir>/<dataset name>/features` as three files per each class: `<classname>_medians.out`, `<classname>_ranklist.out`, `<classname>_scores.out`. 