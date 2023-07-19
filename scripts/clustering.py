from sklearn.cluster import DBSCAN, KMeans
import pandas as pd 
import json
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  
import seaborn as sns

class Clusterer:
    def __init__(self, cluster_type="DBSCAN"):
        self.cluster_type = cluster_type

    def generate_clusters_from_data(self, data, eps=0.5, num_clusters=10):
        '''
        data: (2d array)
        Returns list of cluster labels
        '''
        if self.cluster_type == "DBSCAN":
            clustering = DBSCAN(eps=eps).fit(data)
            cluster_labels = clustering.labels_
            return cluster_labels
        elif self.cluster_type == "KMEANS":
            kmeans = KMeans(n_clusters=num_clusters)
            cluster_labels = kmeans.fit_predict(data)
            return cluster_labels
        else:
            print("Incorrect cluster type given")    
        
    
    def save_tsne(self, perplexity=3, embedding_path="../embeddings/CFD_embeddings.csv"):
        embedding_data = pd.read_csv(embedding_path)
        embeddings = embedding_data['imageEmbedding'].values
        embeddings = np.array([json.loads(embedding) for embedding in embeddings])
        #normalize data
        X = StandardScaler().fit_transform(embeddings)
        #PCA dimensionality reduction to 50
        pca = PCA(n_components=50)
        pca_embeddings = pca.fit_transform(X)
        tsne_embeddings = TSNE(n_components=2, perplexity=perplexity).fit_transform(pca_embeddings)
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:,1], color = '#88c999')
        plt.savefig(f"../files/tsne_{self.cluster_type}_perp={perplexity}_CFD.png")
        return 
        
    def save_clusters(self, eps=0.5, num_clusters=10, embedding_path="../embeddings/CFD_embeddings.csv"):
        embedding_data = pd.read_csv(embedding_path)
        embeddings = embedding_data['imageEmbedding'].values
        embeddings = np.array([json.loads(embedding) for embedding in embeddings])
        #normalize data 
        X = StandardScaler().fit_transform(embeddings)
        cluster_labels = self.generate_clusters_from_data(X, num_clusters=num_clusters)
        image_names = embedding_data['imageName'].values
        #save a dict where key is the cluster label and value is list of indices
        #corresponding to that cluster
        cluster_index_groups = {}
        for i in range(0, len(X)):
            if cluster_labels[i] in cluster_index_groups:
                cluster_index_groups[cluster_labels[i]].append(i)
            else:
                cluster_index_groups[cluster_labels[i]] = [i]
        print(cluster_index_groups)
        return 
        df_list = []
        for cluster_label in cluster_index_groups:
            cluster_image_indices = cluster_index_groups[cluster_label]
            cluster_image_names = image_names[cluster_image_indices]
            df_list.extend(list(zip([cluster_label for _ in range(0, len(cluster_image_names))], cluster_image_names)))
        df = pd.DataFrame(df_list)
        df.to_csv(f"../files/{self.cluster_type}_eps={eps}_CFD.csv")
        return
    
    def save_clusters_eps(self, eps_arr):
        for eps in eps_arr:
            self.save_clusters(eps)



if __name__ == "__main__":
    ClusterGen = Clusterer(cluster_type="KMEANS")
    ClusterGen.save_clusters(num_clusters=50)
    #ClusterGen.save_tsne(perplexity=5)
    #ClusterGen.save_tsne(perplexity=10)
    #ClusterGen.save_tsne(perplexity=20)
    #ClusterGen.save_tsne(perplexity=40)
    #ClusterGen.save_tsne(perplexity=50)
    #ClusterGen.save_clusters_eps(np.linspace(0.001, 0.5, 10))
    


           
                

            





    


    
    

    


     

    