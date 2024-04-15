#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:31:45 2023

@author: msiew, haoranz5
"""

# mmm (diffmodels) noniid

# currently----
# noniid mcm
# iid mcf

# future----
# iid mcf mcf
# noniid mc mc mc
# non iid m c_c100 m c c_100
# non iid m c f, m c f
import numpy as np
import matplotlib.pyplot as plt
from parserplot import ParserArgs
import os
from plotAllocation import plot_allocation
from plotAllocation import simulate_allocation
from plotAllocation import tasklist2clientlist
from plotAllocation import simulate_map
import sys


def AvgAcc1trial(exp_array):
    exp_array_avg = np.mean(exp_array, axis=1)
    return exp_array_avg


def MinAcc1trial(exp_array):
    min_array = np.min(exp_array, axis=1)
    return min_array

def cascadeAcc(exp_array):
    cascade = 1
    for i in range(exp_array.shape[1]):
        cascade *= exp_array[:, i]
    print(cascade.shape)
    return cascade

def diff1trial(exp_array):
    diff = np.max(exp_array, axis=1) - np.min(exp_array, axis=1)
    return diff


def var1trial(exp_array):
    var = np.var(exp_array, axis=1)
    return var



numRounds = 300

def sort_files(files):
    def extract_numbers(file_name):
        parts = file_name.split('_')
        exp_number = int(parts[3].replace('exp', ''))
        algo_number = int(parts[4].replace('algo', '').split('.')[0])
        return algo_number, exp_number

    return sorted(files, key=extract_numbers)

def load_extra_folder(folder_dict, seed=11, header='localAcc_'):
    # each folder should only contain 1 algo result_old
    # get keys of the folder_dict, key is the folder name
    files = []
    for key in folder_dict:
        this_path = os.path.join('./result', key+str(seed))
        files += [os.path.join(this_path, f) for f in os.listdir(this_path) if f.startswith(header)]
    exp_list = []
    for f in files:
        t = np.load(f)
        t = np.where(t <= 0, 0, t)  # t shape: (task_num, numRounds)
        exp_list.append(t)
    exp_array = np.array(exp_list)  # shape 3 5 120

    return exp_array

# read all files
# find all files starting with mcf
# algo_name = ["bayesian", "alpha-fairness", "random", "round robin","optimal_sampling"]


alpha = 1.5
alpha_ms = 4
"""extra_folder = {
    f"2task_nnn_0.01u_a{alpha_ms}_": f"a{alpha_ms}",
    f"2task_nnn_0.01u_AS_clientfair_a{alpha}_": f"AS_CF_a{alpha}",
    f"2task_nnn_0.01u_AS_taskfair_a{alpha}_": f"AS_TF_a{alpha}",
    f"2task_nnn_0.01u_ms_a{alpha_ms}_": f"ms_a{alpha_ms}",
    f"2task_nnn_0.01u_msAS_a{alpha_ms}_": f"msAS_a{alpha_ms}",
    f"2task_nnn_0.01u_OS_clientfair_a{alpha}_": f"OS_CF_a{alpha}",
f"2task_nnn_0.01u_OS_taskfair_a{alpha}_": f"OS_TF_a{alpha}",
"2task_nnn_0.01u_random_": "random",
}
seed_list = [11, 12, 13]"""

extra_folder = {
    #"1task_nnn_u91c0.3_agg_": "test",
    "1task_nnn_u91c0.3_AS_a1_": "AS_a1",
    "1task_nnn_u91c0.3_OS_a1_": "OS_a1",
    "1task_nnn_u91c0.3_AS_a3_": "AS_a3",
    "1task_nnn_u91c0.3_OS_a3_": "OS_a3",
    "1task_nnn_u91c0.3_qFel_a3_": "qFel_a3",
    "1task_nnn_u91c0.3_test_a3_": "test_a3",
    "1task_nnn_u91c0.3_testfixed_a3_": "testfixed_a3",
    "1task_nnn_u91c0.3_testfixed2_a3_": "testfixed2_a3",
    #"1task_nnn_u91c0.3_testEloss_a3_": "testEloss_a3",
    "1task_nnn_u91c0.3_random_": "random"
}
seed_list = [14, 15, 16, 17]

"""extra_folder = {
    "3task_nnn_0.01u_a2_": "a2",
    "3task_nnn_0.01u_AS_clientfair_a2_": "AS_CF_a2",
    "3task_nnn_0.01u_AS_taskfair_a2_": "AS_TF_a2",
    "3task_nnn_0.01u_ms_a4_": "ms_a4",
    "3task_nnn_0.01u_msAS_a4_": "msAS_a4",
    "3task_nnn_0.01u_OS_clientfair_a2_": "OS_CF_a2",
    "3task_nnn_0.01u_OS_taskfair_a2_": "OS_TF_a2",
    "3task_nnn_0.01u_random_": "random"
}
seed_list = [11, 12, 13, 14]"""


"""extra_folder = {
    "4task_nnnn_0.01u_a2_": "a2",
    "4task_nnnn_0.01u_AS_clientfair_a2_": "AS_CF_a2",
    "4task_nnnn_0.01u_AS_taskfair_a2_": "AS_TF_a2",
    "4task_nnnn_0.01u_ms_a4_": "ms_a4",
    "4task_nnnn_0.01u_msAS_a4_": "msAS_a4",
    "4task_nnnn_0.01u_OS_clientfair_a2_": "OS_CF_a2",
    "4task_nnnn_0.01u_OS_taskfair_a2_": "OS_TF_a2",
    "4task_nnnn_0.01u_random_": "random"
}
seed_list = [12, 13]"""

fig_avg = plt.figure()
fig_min = plt.figure()
ax_avg = fig_avg.add_subplot(1, 1, 1)
ax_min = fig_min.add_subplot(1, 1, 1)
for key in extra_folder:
    current_folder = {}
    current_folder[key] = extra_folder[key]
    global_avg_acc = 0
    global_min_acc = 0
    best10_avg = 0
    worst10_avg = 0
    var_avg = 0
    client_var_avg = 0
    curve = []

    for seed in seed_list:
        exp_array = load_extra_folder(current_folder, seed, header='localAcc_')
        # reshape 1 2 40 to 2 40
        exp_array = exp_array.reshape(exp_array.shape[1], exp_array.shape[2])
        path_plot = os.path.join('./result', key)
        algo_name = current_folder.values()
        tasknum = exp_array.shape[0]
        clients_num = exp_array.shape[1]
        # compute average
        exp_array_avg = np.mean(exp_array, axis=0)
        client_var = np.var(exp_array, axis=1)
        client_var_avg += np.mean(client_var)
        # compute minimum
        # sort avg and min
        exp_array_avg = np.sort(exp_array_avg)
        worst10_avg += np.mean(exp_array_avg[:int(clients_num * 0.1)])
        worst10_std = np.std(exp_array_avg[:int(clients_num * 0.1)])
        best10_avg += np.mean(exp_array_avg[-int(clients_num * 0.1):])
        best10_std = np.std(exp_array_avg[-int(clients_num * 0.1):])

        # get global final acc
        exp_array = load_extra_folder(current_folder, seed, header='mcf_i_globalAcc_')
        exp_array = exp_array.reshape(exp_array.shape[1], exp_array.shape[2])
        curve.append(exp_array)

        last = -1
        global_avg_acc += np.mean(exp_array[:, last])
        global_min_acc += np.min(exp_array[:, last])
        var_avg += np.mean(np.var(exp_array))
    global_avg_acc /= len(seed_list)
    global_min_acc /= len(seed_list)
    best10_avg /= len(seed_list)
    worst10_avg /= len(seed_list)
    var_avg /= len(seed_list)
    client_var_avg /= len(seed_list)
    print(f"{next(iter(algo_name)): <10}: \t worst10% {worst10_avg:.3f}, best10% {best10_avg:.3f}, Global acc: {global_avg_acc:.3f}, client_var: {client_var_avg:.3f}.")
    # plot
    # plot global average acc
    #averge the curve
    curve = np.array(curve)
    curve_avg = np.mean(curve, axis=(0,1)).reshape(-1)
    curve_min = np.min(np.mean(curve, axis=0).reshape(tasknum, -1), axis=0).reshape(-1)
    ax_avg.plot(curve_avg, label=next(iter(algo_name)))
    ax_min.plot(curve_min, label=next(iter(algo_name)))


ax_avg.legend()
ax_avg.set_xlabel('Round')
ax_avg.set_ylabel('Global avg acc')
ax_avg.set_title('Global avg acc')

ax_min.legend()
ax_min.set_xlabel('Round')
ax_min.set_ylabel('Global min acc')
ax_min.set_title('Global min acc')
plt.show()

#plt.xlabel('Round')
#plt.ylabel('Global acc')
#plt.title('Global acc')
#plt.legend()
#plt.show()



