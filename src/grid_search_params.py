import os

import keras.regularizers
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import command_args
import create_model
from prepare_data import prepare_dataset


class CustomL1L2Regularizer(keras.regularizers.L1L2):
    def __str__(self):
        as_str = keras.regularizers.serialize(self)
        as_str['config']['l2'] = round(as_str['config']['l2'], 5)
        as_str['config']['l1'] = round(as_str['config']['l1'], 5)
        return str(as_str['config'])


args = command_args.parse()
result_dir = os.path.join(os.pardir, 'result')

X, y = prepare_dataset(args.dataset)
X = X.reshape(X.shape + (1,))

lr_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
reg_list = [
    CustomL1L2Regularizer(l2=0.5), CustomL1L2Regularizer(l2=0.1), CustomL1L2Regularizer(l2=0.05),
    CustomL1L2Regularizer(l2=0.01), CustomL1L2Regularizer(l2=0.005), CustomL1L2Regularizer(l2=0.001),
    CustomL1L2Regularizer(l1=0.5), CustomL1L2Regularizer(l1=0.1), CustomL1L2Regularizer(l1=0.05)
]
param_grid = dict(lr=lr_list, reg=reg_list)


def create_model_wrapper(lr, reg):
    return create_model.binary_model(X, lr, reg)


keras_model = KerasClassifier(build_fn=create_model_wrapper, verbose=0)
score_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
kfold = StratifiedKFold(n_splits=args.splits, shuffle=True)
grid = GridSearchCV(
    estimator=keras_model, param_grid=param_grid, cv=kfold,
    return_train_score=True, scoring=score_list, refit=False
)
grid_result = grid.fit(X, y, epochs=args.epochs, batch_size=args.batch_size)

search_result = pd.DataFrame.from_dict(grid_result.cv_results_)
result_file = os.path.join(result_dir, 'grid_search_result.csv')
search_result.to_csv(result_file)
