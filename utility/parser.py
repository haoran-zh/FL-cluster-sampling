import argparse
import warnings
import os

class ParserArgs(object):
    """
    arguments to be used in the experiment
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.get_general_parser()

    def get_general_parser(self):
        # general settings
        self.parser.add_argument("--seed", type=int, default=13, help="random seed")
        self.parser.add_argument("--round_num", type=int, default=120, help="round number")
        self.parser.add_argument("--C", type=float, default=1.0, help="acive rate: C, numUsersSel=C*numClients")
        self.parser.add_argument("--class_ratio", nargs='*', type=float, default=[0.3, 0.3, 0.3, 0.3, 0.3], help="list of class ratios for noniid task")
        self.parser.add_argument("--local_epochs", nargs='*', type=int, default=[5, 5, 5, 5, 5],
                                 help="local epochs for each task")
        self.parser.add_argument("--num_clients", type=int, default=30, help="number of clients")
        self.parser.add_argument("--exp_num", type=int, default=1, help="experiment number. repeat the same experiment several times")
        self.parser.add_argument("--powerfulCNN", action="store_true", help="Decide if use powerful CNN for EMNIST")
        self.parser.add_argument("--iid_type", nargs='*', default=['noniid', 'noniid', 'noniid', 'iid', 'iid'], help="List of type_iid")
        self.parser.add_argument("--task_type", nargs='*', default=['mnist', 'cifar10', 'fashion_mnist', 'emnist', 'cifar10'], help="List of task types")
        self.parser.add_argument("--alpha", type=float, default=3.0, help="alpha value, alpha-fairness")
        self.parser.add_argument("--notes", type=str, default='', help="notes to add on the folder name")
        self.parser.add_argument("--bayes_decay", type=float, default=1.0, help="Decay factor for bayesian method")
        self.parser.add_argument("--algo_type", nargs='*', default=['bayesian', 'proposed','random','round_robin'], help="List of algorithms")
        self.parser.add_argument("--data_ratio", type=float, default=1.0, help="data points num=default_num*data_ratio")
        self.parser.add_argument("--cpumodel", action="store_true", help="store model in cpu")
        self.parser.add_argument("--alpha_loss", action="store_true", help="use alpha-fairness loss function, remember to adjust learning rate")
        self.parser.add_argument("--optimal_sampling", action="store_true", help="use optimal sampling")
        self.parser.add_argument("--insist", action="store_true", help="if insist, then experiment will be conducted even if the folder exists")
        self.parser.add_argument("--lr", type=float, default=1, help="learning rate")
        self.parser.add_argument("--validation", action="store_true", help="use validation set instead of test set")
        self.parser.add_argument("--unbalance", nargs='*', type=float, default=[1.0, 1.0], help="unbalance[0]% clients will get unbalance[1]% data")
        self.parser.add_argument("--fairness", type=str, default='notfair', help="fairness type")
        self.parser.add_argument("--L", type=float, default=1/0.02, help="learning rate")
        self.parser.add_argument("--approx_optimal", action="store_true", help="act approx optimal")
        #self.parser.add_argument("--aggregation_fair", action="store_true", help="act aggregation fairness")
        #self.parser.add_argument("--equalP", action="store_true", help="make P equal")
        #self.parser.add_argument("--enlarge", action="store_true", help="make P large")
        #self.parser.add_argument("--equalP2", action="store_true", help="make P equal, and use optimal Prob")




    def get_args(self):
        args = self.parser.parse_args()
        return args