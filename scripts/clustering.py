from sklearn.cluster import DBSCAN, KMeans
import pandas as pd 
import json
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  
import seaborn as sns
from pathlib import Path

class Clusterer:
    def __init__(self, cluster_type="DBSCAN", embedding_path='../embeddings/facenet512_CFD_embeddings.csv'):
        self.cluster_type = cluster_type
        self.embedding_path = embedding_path

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
        
    
    def save_tsne(self, perplexity=3):
        embedding_data = pd.read_csv(self.embedding_path)
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
        
    def save_clusters(self, eps=0.5, num_clusters=10):
        '''
        Generate clusters and save the cluster label and file name df to csv
        '''
        if Path(self.embedding_path).suffix == ".csv":
            embedding_df = pd.read_csv(self.embedding_path)
            embeddings = embedding_df['embeddings'].values
            embeddings = np.array([json.loads(embedding) for embedding in embeddings]) 
        elif Path(self.embedding_path).suffix == ".pkl":
            embedding_df = pd.read_pickle(self.embedding_path)
            embeddings = embedding_df['embeddings'].to_numpy()
            embeddings = np.stack(embeddings)
            print(f'Embeddings shape {embeddings.shape}')
        else:
            print(f"File {self.embedding_path} is not of form csv or pkl")
        #normalize data 
        X = StandardScaler().fit_transform(embeddings)
        cluster_labels = self.generate_clusters_from_data(X, num_clusters=num_clusters)
        image_names = embedding_df['image_names'].values
        
        #save a dict where key is the cluster label and value is list of indices
        #corresponding to that cluster
        cluster_index_groups = {}
        for i in range(0, len(X)):
            if cluster_labels[i] in cluster_index_groups:
                cluster_index_groups[cluster_labels[i]].append(i)
            else:
                cluster_index_groups[cluster_labels[i]] = [i]

        df_list = []
        for cluster_label in cluster_index_groups:
            cluster_image_indices = cluster_index_groups[cluster_label]
            cluster_image_names = image_names[cluster_image_indices]
            df_list.extend(list(zip([cluster_label for _ in range(0, len(cluster_image_names))], cluster_image_names)))
        df = pd.DataFrame(df_list)
        if self.cluster_type == "KMEANS":
            csv_save_name = f"../files/{self.cluster_type}_k={num_clusters}_{Path(self.embedding_path).stem}.csv"
        if self.cluster_type == "DBSCAN":
            csv_save_name = f"../files/{self.cluster_type}_eps={eps:.2f}_{Path(self.embedding_path).stem}.csv"
        df.to_csv(csv_save_name)
        return
    
    def save_clusters_eps(self, eps_arr, k_arr):
        '''
        Generate multiple cluster label and filename csv's using an array of parameters for the clustering 
        '''
        if eps_arr is not None:
            for eps in eps_arr:
                self.save_clusters(eps)
        if k_arr is not None:
            for k in k_arr:
                self.save_clusters(num_clusters=k)
        return 



if __name__ == "__main__":
    ClusterGen = Clusterer(cluster_type="KMEANS", embedding_path="../embeddings/male_5000_5-5_nonan.pkl")
    ClusterGen.save_clusters(num_clusters=10)
    #ClusterGen.save_tsne(perplexity=5)
    #ClusterGen.save_tsne(perplexity=10)
    #ClusterGen.save_tsne(perplexity=20)
    #ClusterGen.save_tsne(perplexity=40)
    #ClusterGen.save_tsne(perplexity=50)
    #ClusterGen.save_clusters_eps(np.linspace(0.001, 0.5, 10))
    


           
                

            





    


    
    

    


     

    