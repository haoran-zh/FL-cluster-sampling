from .utils.constants import *
from .utils.utils import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

class Cluster:
    def __init__(self, clients, num_clusters=5):
        self.clients = clients
        self.num_clusters = num_clusters
        # self.cluster()
    
    def create_latent_space(self, data_type="gradient_norms", dim_method="pca", data=None):
        # pick data type
        if data_type == "gradient_norms":
            self.data = np.array([c.gradient_norms for c in self.clients])
        elif data_type == "weights":
            self.data = np.array([c.flattened_weights for c in self.clients])
        elif data_type == "gradients":
            self.data = np.array([c.flattened_gradients for c in self.clients])
        elif data_type == "custom":
            assert data is not None, "Data must be provided for custom data type!"
            self.data = data
        else:
            raise ValueError(f"Data type {data_type} not supported!")
        # pick dimension reduction method
        if dim_method == "pca":
            pca = PCA(n_components=2)
            self.latent_space = pca.fit_transform(self.data) 
        elif dim_method == "tsne":
            tsne = TSNE(n_components=2, learning_rate='auto',
                          init='random', perplexity=3)
            self.latent_space = tsne.fit_transform(self.data)
        elif dim_method == "autoencoder":
            pass
        else:
            raise ValueError(f"Method {dim_method} not supported!")
        
    def cluster_latent_space(self, cluster_method="kmeans"):
        if cluster_method == "kmeans":
            self.kmeans = KMeans(n_clusters=self.num_clusters, n_init='auto')
            self.kmeans.fit(self.latent_space)
            self.cluster_labels = self.kmeans.labels_
        elif cluster_method == "gmm":
            self.gmm = GaussianMixture(n_components=self.num_clusters)
            self.cluster_labels = self.gmm.fit_predict(self.latent_space)
        else:
            raise ValueError(f"Cluster method {cluster_method} not supported!")
    
    def cluster(self, method="non_cosine", data_type="gradient_norms", dim_method="pca", cluster_method="kmeans"):
        #for c in self.clients:
        #    c.get_trained_weights()
        # try to use gradient similarity
        if method == "cosine":
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
            self.create_latent_space(data_type=data_type, dim_method=dim_method)
            assert self.latent_space is not None, "Failed creating latent space."
            self.cluster_latent_space(cluster_method=cluster_method)
            assert self.cluster_labels is not None, "Failed clustering latent space."
            # self.plot()
        
    def plot(self):
        plt.figure(figsize=(10, 10))
        print("Silhouette score:", silhouette_score(self.latent_space, self.cluster_labels))
        for i in range(self.num_clusters):
            plt.scatter(self.latent_space[self.cluster_labels == i, 0], 
                        self.latent_space[self.cluster_labels == i, 1], 
                        label=f'Cluster {i}')
        for i, (x, y) in enumerate(self.latent_space):
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