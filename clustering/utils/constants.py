# experiment settings
random_seed = 18786

type_iid=["noniid"]
task_type = ["mnist"]
num_clients = 20
class_ratio=[0.21]
active_rate = 0.2

class ARGS:
    def __init__(self):
        global num_clients
        self.num_clients = num_clients
        self.unbalance = [1.0, 1.0]  # [0]% of clients have [1]% of data
args = ARGS()