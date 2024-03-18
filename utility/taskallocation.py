import numpy as np

def get_task_idx(num_tasks,
                 num_clients,
                 algorithm_name,
                 normalization,
                 tasks_weight,
                 global_accs, beta,
                 # NEW: dec 6 2023
                 chosen_clients, # looks here is unfinished
                 allocation_history,
                 args
                 ):
    mixed_loss = [1.] * num_tasks  # create a list of num_tasks elements, each element is 1
    for task_idx in range(num_tasks):
        if normalization == 'accuracy':
            mixed_loss[task_idx] *= tasks_weight[task_idx] * \
                                    np.power(100 * (1. - global_accs[task_idx]), beta - 1)

    if algorithm_name == 'proposed':
        # print('prop')
        probabilities = np.zeros((num_tasks))
        for task_idx in range(num_tasks):
            probabilities[task_idx] = mixed_loss[task_idx] / (np.sum(mixed_loss))
        # print(probabilities)
        # to double check!!!
        # 1,2,3
        # client 1: P(alpha-fairness), task 1,2,3
        # client 2:
        return list(np.random.choice(np.arange(0, num_tasks), num_clients, p=probabilities))
    elif algorithm_name == 'bayesian':
        past_counts = np.zeros((num_clients, num_tasks))  # num_clients needs to be changed to len(chosen_clients) in the future
        history_array = np.array(allocation_history) # shape: rounds * num_clients
        d = args.bayes_decay
        round_num = len(allocation_history)
        for client_idx in range(num_clients):
            for task_idx in range(num_tasks):
                for i in range(round_num):
                    #  past_counts[client_idx, task_idx] += (d ** (round_num-i-1)) * (np.sum(history_array[i, client_idx] == task_idx) + 1)
                    past_counts[client_idx, task_idx] += (d ** (round_num - i - 1)) * (
                                np.sum(history_array[i, client_idx] == task_idx)+1)
        # normalization
        past_counts = past_counts / np.sum(past_counts, axis=1, keepdims=True)
        future_expect = 1 - past_counts
        # past_counts cannot exceed 0.5
        #------------
        #past_counts = np.minimum(past_counts, 0.45)
        #future_expect = 1/2 * np.log((1-past_counts)/past_counts)
        # ------------
        if args.bayes_exp:
            future_expect = np.exp(future_expect)
        P_client_task = future_expect / np.sum(future_expect, axis=1, keepdims=True)

        P_task = np.zeros((num_tasks))
        for task_idx in range(num_tasks):
            P_task[task_idx] = mixed_loss[task_idx] / (np.sum(mixed_loss))

        P_task_client = np.zeros((num_clients, num_tasks))

        for client_idx in range(num_clients):
            for task_idx in range(num_tasks):
                P_task_client[client_idx, task_idx] = P_task[task_idx] * P_client_task[client_idx, task_idx]
        # normalization
        P_task_client = P_task_client / np.sum(P_task_client, axis=1, keepdims=True)

        allocation_result = np.zeros(num_clients, dtype=int)
        # include the probability that this client is not selected, extra column. -1 if not selected. unfinished!!!!!!
        for client_idx in range(num_clients):
            if client_idx not in chosen_clients:
                allocation_result[client_idx] = -1
            else:
                allocation_result[client_idx] = np.random.choice(np.arange(0, num_tasks), p=P_task_client[client_idx])
        allocation_result = allocation_result.tolist()
        return allocation_result, P_task_client


    elif algorithm_name == 'random':
        # clients_task=[]
        # for i in range(len(chosen_clients)):
        #     task=np.random.randint(0,num_tasks)
        #     clients_task.append(task)

        # return clients_task #
        return list(np.random.randint(0, num_tasks, num_clients, dtype=int))


def get_task_id_RR(num_tasks, totNumCl,
                   num_clients,
                   algorithm_name,
                   normalization,
                   tasks_weight,
                   global_accs, beta, firstIndRR, rr_taskAlloc, rr_chosen_clients):
    # numUsersSel=num_clients
    # numTasks=len(tasks_weight)
    # #resetting
    # firstIndRR=(firstIndRR+numUsersSel)%totNumCl
    # print('firstIndRR', firstIndRR)
    # rr_taskAlloc[rr_chosen_clients]=(rr_taskAlloc[rr_chosen_clients]+1)%numTasks
    # print('new allocation vec')
    # print(rr_taskAlloc)

    # NEW: dec 6 2023
    # NEW! to make the available clients the same.
    rr_taskAlloc = (rr_taskAlloc + 1) % num_tasks
    clients_task = []
    for i in range(len(rr_chosen_clients)):
        client = rr_chosen_clients[i]
        task = rr_taskAlloc[client]
        clients_task.append(task)

    # clients_task=rr_taskAlloc[int(firstIndRR):int(firstIndRR+numUsersSel)]
    # rr_chosen_clients=np.arange(totNumCl)[int(firstIndRR):int(firstIndRR+numUsersSel)]
    # print('total num cl', totNumCl)
    # print('rr chosen clients')
    # print(rr_chosen_clients)
    # print('task allocation')
    # print(clients_task)

    return clients_task, rr_chosen_clients, firstIndRR, rr_taskAlloc
       



    