import os
import matplotlib.pyplot as plt
import pandas as pd

# stat_name_list = ['acc', 'val_acc']
stat_name_list = ['val_recall', 'val_precision']
index_list = range(10)
result_dirpath = os.path.join(os.pardir, 'result', '400e_usual')

# collect data
result_df_list = []
for i in index_list:
    i_result_filepath = os.path.join(result_dirpath, 'cirrhosis_hist' + str(i) + '.csv')
    i_df = pd.read_csv(i_result_filepath, sep=';')
    result_df_list.append(i_df)

# display data
plt.figure(figsize=(14, 8))
for i, i_df in enumerate(result_df_list):
    plt.subplot(2, 5, i + 1)
    plt.plot(i_df[stat_name_list[0]], 'b-', label=stat_name_list[0])
    plt.plot(i_df[stat_name_list[1]], 'c-', label=stat_name_list[1])
    plt.legend()

plt.tight_layout()
plt.show()
