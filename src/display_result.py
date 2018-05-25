import json
import os
import matplotlib.pyplot as plt

stat_name_list = ['acc', 'val_acc']
index_list = range(10)
result_dirpath = os.path.join(os.pardir, 'result', '1000')

# collect data
result_list = []
for i in index_list:
    i_result_filepath = os.path.join(result_dirpath, 'cirrhosis_hist' + str(i) + '.txt')
    with open(i_result_filepath) as result_file:
        result_dict = json.load(result_file)
        result_list.append(result_dict)

# display data
plt.figure(figsize=(14, 8))
for i, result_dict in enumerate(result_list):
    plt.subplot(2, 5, i + 1)
    plt.plot(result_dict[stat_name_list[0]], 'b-', label=stat_name_list[0])
    plt.plot(result_dict[stat_name_list[1]], 'c-', label=stat_name_list[1])
    plt.legend()

plt.tight_layout()
plt.show()
