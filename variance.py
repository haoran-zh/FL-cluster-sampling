import numpy as np
import matplotlib.pyplot as plt
import os

def sort_files(files):
    def extract_numbers(file_name):
        parts = file_name.split('_')
        exp_number = int(parts[3].replace('exp', ''))
        algo_number = int(parts[4].replace('algo', '').split('.')[0])
        return algo_number, exp_number

    return sorted(files, key=extract_numbers)

algo_num = 3
paths = []
# list all the paths
# each path is a folder containing 3 algo results
# each folder should exist:
# (mcf_..._exp0_algo0.npy)
# (mcf_..._exp0_algo1.npy)
# (mcf_..._exp0_algo2.npy)
paths.append(os.path.join('./result', "10task_iiiiiiiiii_exp1C1c20-cpu-seed10"))
paths.append(os.path.join('./result', "10task_iiiiiiiiii_exp1C1c20-cpu-seed10"))
paths.append(os.path.join('./result', "10task_iiiiiiiiii_exp1C1c20-cpu-seed11"))
paths.append(os.path.join('./result', "10task_iiiiiiiiii_exp1C1c20-cpu-seed12"))
exp_seeds_array = []
for path_plot in paths:
    files = [f for f in os.listdir(path_plot) if f.startswith('mcf')]
    files = sort_files(files)
    # skip the bayesian
    if 'algo3' in files[-1]:
        files = files[1:]

    exp_list = []
    for f in files:
        t = np.load(os.path.join(path_plot, f))
        t = np.where(t <= 0, 0, t)
        exp_list.append(t)
    exp_array = np.array(exp_list)  # shape 3 5 120
    exp_num = int(exp_array.shape[0] / algo_num)


    if exp_num > 1:
        aver_list = []
        # compute average. example: 16 5 120 average to 4 5 120
        for i in range(exp_num):
            average = np.mean(exp_array[i*exp_num:(i+1)*exp_num], axis=0)
            aver_list.append(average)
        exp_array = np.array(aver_list)
    exp_seeds_array.append(exp_array)

exp_seeds_array = np.array(exp_seeds_array)
print(exp_seeds_array.shape) # 4seeds 3algo 6tasks 120rounds

"""#plot std
std_array = np.std(exp_seeds_array, axis=2)
std_array = np.mean(std_array, axis=0) # get shape 3algo 120rounds"""
# plot variance
seeds_var_array = np.var(exp_seeds_array, axis=2) # get shape 4seeds 3algo 120rounds

# compute error bars
errors_algo1 = [[np.min(np.mean(seeds_var_array[:,0], axis=1), axis=0), np.max(np.mean(seeds_var_array[:,0], axis=1), axis=0)],
                [np.min(seeds_var_array[:,0,10]), np.max(seeds_var_array[:,0,10])],
                [np.min(seeds_var_array[:,0,60]), np.max(seeds_var_array[:,0,60])],
                [np.min(seeds_var_array[:,0,119]), np.max(seeds_var_array[:,0,119])]]
errors_algo2 = [[np.min(np.mean(seeds_var_array[:,1], axis=1), axis=0), np.max(np.mean(seeds_var_array[:,1], axis=1), axis=0)],
                [np.min(seeds_var_array[:, 1, 10]), np.max(seeds_var_array[:, 1, 10])],
                [np.min(seeds_var_array[:, 1, 60]), np.max(seeds_var_array[:, 1, 60])],
                [np.min(seeds_var_array[:, 1, 119]), np.max(seeds_var_array[:, 1, 119])]]
errors_algo3 = [[np.min(np.mean(seeds_var_array[:,2], axis=1), axis=0), np.max(np.mean(seeds_var_array[:,2], axis=1), axis=0)],
                [np.min(seeds_var_array[:, 2, 10]), np.max(seeds_var_array[:, 2, 10])],
                [np.min(seeds_var_array[:, 2, 60]), np.max(seeds_var_array[:, 2, 60])],
                [np.min(seeds_var_array[:, 2, 119]), np.max(seeds_var_array[:, 2, 119])]]

mean_var_array = np.mean(seeds_var_array, axis=2) # get shape 4seeds 3algo 120rounds
mean_var_array = np.mean(mean_var_array, axis=0)
var_array = np.mean(seeds_var_array, axis=0)

labels = ['avg. var','round=10' ,'round=60', 'round=120']


algo1 = [mean_var_array[0], var_array[0][10],var_array[0][60], var_array[0][119]]
algo2 = [mean_var_array[1], var_array[1][10],var_array[1][60], var_array[1][119]]
algo3 = [mean_var_array[2], var_array[2][10],var_array[2][60], var_array[2][119]]

# finish the computation of error bars
errors_algo1 = [[abs(algo1[i]-errors_algo1[i][0]),abs(algo1[i]-errors_algo1[i][1])] for i in range(4)]
errors_algo2 = [[abs(algo2[i]-errors_algo2[i][0]),abs(algo2[i]-errors_algo2[i][1])] for i in range(4)]
errors_algo3 = [[abs(algo3[i]-errors_algo3[i][0]),abs(algo3[i]-errors_algo3[i][1])] for i in range(4)]



bar_width = 0.2
index = np.arange(len(labels))
fig, ax = plt.subplots()
bar1 = ax.bar(index, algo1, bar_width, yerr=np.array(errors_algo1).T, capsize=5, label='Alpha-fair Client-Task Allocation',alpha=0.5, color='blue')
bar2 = ax.bar(index + bar_width, algo2, bar_width, yerr=np.array(errors_algo2).T, capsize=5, label='Random Client-Task Allocation',color='orange')
bar3 = ax.bar(index + 2*bar_width, algo3, bar_width, yerr=np.array(errors_algo3).T, capsize=5, label='Round Robin Client-Task Allocation', color='green')

# Adding labels, title, and legend
#ax.set_xlabel('The Number of Slots (N)', fontsize=14)
ax.set_ylabel('Variance', fontsize=14)
ax.set_title('Variance over the algorithms, 10 tasks', fontsize=14)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(labels, fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig('variance.png')
plt.clf()