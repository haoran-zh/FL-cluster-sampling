#!/bin/bash
# 1 task experiments
# project: cluster-based sampling
# --L: 1/L is the learning rate.
# --unbalance: the unbalance level of the dataset
# For each client, it will have 400~500 data points by default.
# --unbalance 0.9 0.1 means 90% (0.9) clients will have 40~50 data points (400*0.1~500*0.1).
# --fairness: we don't need to consider this.
# --alpha: the true loss function will be f^alpha. In this project, we set alpha=1.
# --notes: the notes of the experiment. will become part of the experiment folder name.
# --C: class rate, control the non-iid level. Each client will only has data points from C*total_classes classes. Example: C=0.2, total_clases=10, then each client only has 2 classes data.
# --insist: overwrite the existing folder.
seedlist=(14 15 16 17 18 19 20 21)
for sd in "${seedlist[@]}"; do
# random, sampling is uniform
python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 1 --notes u91c0.3_random_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type random --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
# optimal sampling with loss value
python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 1 --notes u91c0.3_AS_a1_$sd --approx_optimal --alpha_loss --C 0.2 --num_clients 40 --class_ratio 0.3 --iid_type noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 --round_num 300 --insist
# optimal sampling with gradient norm
python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 1 --notes u91c0.3_OS_a1_$sd --optimal_sampling --alpha_loss --C 0.2 --num_clients 40 --class_ratio 0.3 --iid_type noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 --round_num 300 --insist
done