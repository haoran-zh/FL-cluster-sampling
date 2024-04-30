from .utils.constants import *
from .utils.utils import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, clients, num_clusters=5):
        self.clients = clients
        self.num_clusters = num_clusters
        # self.cluster()
        
    def cluster(self, method="gradient_norm"):
        #self.client_weights = np.array([c.flattened_weights for c in self.clients])
        #self.client_gradients = np.array([c.flattened_gradients for c in self.clients])
        #for c in self.clients:
        #    c.get_trained_weights()

        # try to use gradient similarity
        if method == "cosine":
            from sklearn.cluster import SpectralClustering
            gradient_similarity = np.ones((len(self.clients), len(self.clients)))
            for i in range(len(self.clients)):
                for j in range(i+1, len(self.clients)):
                    gradient_i = self.clients[i].gradients
                    gradient_j = self.clients[j].gradients
                    gradient_similarity[i, j] = cosine_similarity(gradient_i, gradient_j)
                    gradient_similarity[j, i] = gradient_similarity[i, j]
            print(gradient_similarity)
            clustering = SpectralClustering(n_clusters=self.num_clusters,
                                    affinity='precomputed',
                                    assign_labels='kmeans',
                                    random_state=0).fit(gradient_similarity)
            self.cluster_labels = clustering.labels_
        else:
            self.client_norms = np.array([c.gradient_norms for c in self.clients])
            self.data = self.client_norms
            self.pca = PCA(n_components=2)
            self.client_pca = self.pca.fit_transform(self.data)
            self.kmeans = KMeans(n_clusters=self.num_clusters, n_init='auto')
            self.kmeans.fit(self.client_pca)
            
            self.cluster_labels = self.kmeans.labels_
            self.cluster_centers = self.kmeans.cluster_centers_
            # self.plot()
        
    def plot(self):
        plt.figure(figsize=(10, 10))
        for i in range(self.num_clusters):
            plt.scatter(self.client_pca[self.cluster_labels == i, 0], 
                        self.client_pca[self.cluster_labels == i, 1], 
                        label=f'Cluster {i}')
        for i, (x, y) in enumerate(self.client_pca):
            plt.annotate(f'Client {i}; labels {self.clients[i].non_iid_labels}', 
                         (x, y), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center')
        plt.legend()
        plt.show()
    
    def get_result(self):
        #return pd.DataFrame({"cluster_labels": self.cluster_labels, 
        #                     "client_index": [c.client_index for c in self.clients]})
        cluster_clients_list = []
        for cluster in range(self.num_clusters):
            cluster_clients_list.append([])
            # initialize cluster_clients_list

        for c in self.clients:
            c_index = c.client_index
            cluster = self.cluster_labels[c_index]
            cluster_clients_list[cluster].append(c)
        self.result = cluster_clients_list
        return self.result