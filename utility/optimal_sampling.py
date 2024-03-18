import torch
import numpy as np
import random

def get_gradient_norm(weights_this_round, weights_next_round, args):
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_this_round[name] - weights_next_round[name]).cpu() for name in weights_this_round}
    # Calculate the L2 norm of the weight differences
    norm = sum(torch.norm(diff, p=2) ** 2 for diff in weight_diff.values()) ** 0.5 / args.lr
    norm.item()
    return norm.item(), weight_diff


def power_gradient_norm(gradient_norm, tasks_local_training_loss, args, all_data_num):
    alpha = args.alpha
    task_num = len(args.task_type)
    gradient_norm = np.array(gradient_norm)
    tasks_local_training_loss = np.array(tasks_local_training_loss)
    gradient_norm_power = gradient_norm * np.power(tasks_local_training_loss, (alpha - 1)) * alpha
    return gradient_norm_power


def get_sampling_clusters(chosen_clients, clients_task, all_data_num, gradient_record, args):
    # We may need to change the inputs.
    
    # chosen_clients is a list, including all active clients indices for the current round
    # example: chosen_clients = [19, 20, 1, 0, 33] Current round active clients are client 19, client 20, client 1, client 0, client 33
    # clients_task is a list, showing the task index for each client in chosen_clients
    # example: clients_task = [1,1,0,0,2]. Then client 19 will train task1, client 20 will train task1...
    # We are considering doing standard FL with only 1 task, so the elements in this list will only be 0. [0,0,0,0,0]
    # all_data_num[t][k] is the number of data points of task t on client k
    # gradient_record[t][k] is the gradient norm of task t on client k
    # TODO: clustering-based sampliing algorithm
    # 1. Decide the number of clusters
    # 2. For each cluster, do the importance sampling. (slightly modify the get_optimal_sampling function will be enough)
    # return
    # client_clusters: client_clusters[i] include the indices of clients in the i-th cluster.
    pass




def get_optimal_sampling(chosen_clients, clients_task, all_data_num, gradient_record, args): # gradient record is gradient norm
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    if type(clients_task) == list:
        clients_task = np.array(clients_task)
    sample_num = len(chosen_clients)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])

    all_gradients = gradient_record.copy()


    for task_index in range(tasks_num):
        for client_index in range(all_clients_num):
        # from U to U~ in the paper
            all_gradients[task_index][client_index] *= all_data_num[task_index][client_index] / np.sum(
            all_data_num[task_index])

    client_gradients_sumTasks = np.zeros(all_clients_num) # this is M_i in the proof
    for client_index in range(all_clients_num):
        for task_index in range(tasks_num):
            client_gradients_sumTasks[client_index] += all_gradients[task_index][client_index]

    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(client_gradients_sumTasks)

    n = all_clients_num
    m = sample_num

    l = n - m + 1
    best_l = l
    if m == 0: # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
        best_l = n

    while True:
        l += 1
        if l > n:
            break
        # sum the first l smallest gradients
        sum_upto_l = sum(client_gradients_sumTasks[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / client_gradients_sumTasks[sorted_indices[l-1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_s_i = np.zeros((tasks_num, all_clients_num))
    sum_upto_l = sum(client_gradients_sumTasks[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = all_gradients[task_index][sorted_indices[i]] / client_gradients_sumTasks[sorted_indices[i]]
        else:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = (m + best_l - n) * all_gradients[task_index][sorted_indices[i]] / sum_upto_l

    allocation_result = np.zeros(all_clients_num, dtype=int)
    for client_idx in range(all_clients_num):
        if abs(1-np.sum(p_s_i[:, client_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - np.sum(p_s_i[:, client_idx])
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num+1)
        p_client[0] = p_not_choose
        p_client[1:] = p_s_i[:, client_idx]
        allocation_result[client_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client)
    allocation_result = allocation_result.tolist()
    clients_task = [s for s in allocation_result if s != -1]
    chosen_clients = [i for i in range(len(allocation_result)) if allocation_result[i] != -1]
    # get p_dict
    p_dict = []
    active_rate = len(chosen_clients)/all_clients_num

    for task_index in range(tasks_num):
        p_dict.append([p_s_i[task_index][i] for i in range(all_clients_num) if allocation_result[i] == task_index])

    return clients_task, p_dict, chosen_clients

def get_clients_num_per_task(clients_task, tasks_num):
    clients_num_per_task = [0] * tasks_num # list with length of tasks_num
    for task_index in range(tasks_num):
        # count the number of clients for each task
        clients_num_per_task[task_index] = clients_task.tolist().count(task_index)
    return clients_num_per_task


import cvxpy as cp

def optimal_solver(client_num, task_num, all_gradients, ms_list):
    N = client_num  # Number of clients
    S = task_num    # Number of tasks
    U = np.array(all_gradients).reshape(task_num, client_num)  # Gradient record reshaped
    ms = np.array(ms_list)  # List of ms values
    # if ms exist 0, then the problem is infeasible
    # if 0 in ms, adjust ms to make it feasible
    if 0 in ms:
        for i in range(len(ms)):
            if ms[i] == 0:
                ms[i] = 1
    # Define the variable to solve for
    p = cp.Variable((S, N), nonneg=True)
    # print("curvature of p", p.curvature)

    # Objective function
    # print("curvature of U", U.curvature)
    U2 = cp.square(U)
    objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))

    # Constraints
    constraints = []
    # Sum of p for each client across tasks <= 1
    constraints += [cp.sum(p, axis=0) <= 1]
    # Sum of p for each task across clients <= ms
    constraints += [cp.sum(p, axis=1) <= ms]
    # p >= 0 (implicitly satisfies p <= 1 due to the constraints above)
    constraints += [p >= 1e-10]
    constraints += [p <= 1]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(max_iters=800000)

    # Check if the problem is solved successfully
    if problem.status not in ["infeasible", "unbounded"]:
        # Successfully solved
        p_optimal = p.value
    else:
        print("Optimization failed:", problem.status)
        p_optimal = np.ones((S, N)) / S  # Return a default value or handle the failure appropriately
    return p_optimal


def get_optimal_sampling_cvx(chosen_clients, clients_task, all_data_num, gradient_record):
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    if type(clients_task) == list:
        clients_task = np.array(clients_task)
    sample_num = len(chosen_clients)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])

    ms_list = get_clients_num_per_task(clients_task, tasks_num)

    all_gradients = gradient_record.copy()

    for task_index in range(tasks_num):
        for client_index in range(all_clients_num):
        # from U to U~ in the paper
            all_gradients[task_index][client_index] *= all_data_num[task_index][client_index] / np.sum(
            all_data_num[task_index])

    p_optimal = optimal_solver(client_num=all_clients_num, task_num=tasks_num, all_gradients=all_gradients, ms_list=ms_list)
    p_s_i = p_optimal

    allocation_result = np.zeros(all_clients_num, dtype=int)
    for client_idx in range(all_clients_num):
        if abs(1 - np.sum(p_s_i[:, client_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - np.sum(p_s_i[:, client_idx])
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num + 1)
        p_client[0] = p_not_choose
        p_client[1:] = p_s_i[:, client_idx]
        try:
            allocation_result[client_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client)
        except:
            allocation_result[client_idx] = np.random.choice(np.arange(-1, tasks_num), p=np.ones(tasks_num+1)/(1+tasks_num))

    allocation_result = allocation_result.tolist()
    clients_task = [s for s in allocation_result if s != -1]
    chosen_clients = [i for i in range(len(allocation_result)) if allocation_result[i] != -1]
    # get p_dict
    p_dict = []
    for task_index in range(tasks_num):
        p_dict.append([p_s_i[task_index][i] for i in range(all_clients_num) if allocation_result[i] == task_index])
    return clients_task, p_dict, chosen_clients


def aggregation_fair(loss_af_aggregation, loss_bf_aggregation):
    loss_diff = loss_af_aggregation - loss_bf_aggregation
    # we want loss_diff to be small
    # if loss_diff is negative, change to 0
    loss_diff = np.maximum(loss_diff, 1e-6)
    return loss_diff

