from sklearn.cluster import DBSCAN, KMeans
from sklearn_extra.cluster import KMedoids
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  
import seaborn as sns
from pathlib import Path

class Clusterer:
    def __init__(self, cluster_type="KMEANS", embedding_path='../embeddings/tmdb_female_5000.pkl'):
        self.cluster_type = cluster_type
        self.embedding_path = embedding_path
        embedding_df = pd.read_pickle(embedding_path)
        embeddings = embedding_df['embeddings'].to_numpy()
        embeddings = np.stack(embeddings)
        self.embeddings = embeddings
        image_names = embedding_df['image_names'].to_numpy()
        self.image_names = image_names
        #array where A[i] is the tmdb_id of the ith embedding
        tmdb_id_arr = [int(image_name.split("_")[-1]) for image_name in image_names] 
        self.tmdb_id_arr = tmdb_id_arr

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
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
            cluster_labels = kmeans.fit_predict(data)
            return cluster_labels
        elif self.cluster_type == "KMEDOIDS":
            kmedoids = KMedoids(n_clusters=num_clusters, init="k-medoids++").fit(data) 
            cluster_labels = kmedoids.labels_ 
            center_idxs = kmedoids.medoid_indices_
            center_ids = [self.tmdb_id_arr[idx] for idx in center_idxs]
            return cluster_labels, center_ids 
        else:
            print("Incorrect cluster type given")    
        
    def ret_clusters_dict(self, num_clusters):
        '''
        Returns a dict where the key is the cluster label and value is a list of tmdb IDS
        '''
        #normalize data 
        X = StandardScaler().fit_transform(self.embeddings)
        if self.cluster_type == "KMEDOIDS":
            cluster_labels, center_ids = self.generate_clusters_from_data(X, num_clusters=num_clusters)
            center_ids = np.array(center_ids)
        else:
            cluster_labels = self.generate_clusters_from_data(X, num_clusters=num_clusters)
        
        #key- cluster label. value- list of tmdb ID's in that cluster
        cluster_index_groups = {}
        for i in range(0, len(X)):
            if cluster_labels[i] in cluster_index_groups:
                cluster_index_groups[cluster_labels[i]].append(self.tmdb_id_arr[i])
            else:
                cluster_index_groups[cluster_labels[i]] = [self.tmdb_id_arr[i]]
        return cluster_index_groups 
    
    def save_clusters(self, eps=0.5, num_clusters=10):
        '''
        Generate clusters. Saves a csv where the first col is the cluster label and second col is the image_name (including tmdb ID)
        Returns: saved csv file path 
        '''
        #normalize data 
        X = StandardScaler().fit_transform(self.embeddings)
        if self.cluster_type == "KMEDOIDS":
            cluster_labels, center_ids = self.generate_clusters_from_data(X, num_clusters=num_clusters)
            center_ids = np.array(center_ids)
        else:
            cluster_labels = self.generate_clusters_from_data(X, num_clusters=num_clusters)
        
        #key- cluster label. value- list of embedding idxs
        cluster_index_groups = {}
        for i in range(0, len(X)):
            if cluster_labels[i] in cluster_index_groups:
                cluster_index_groups[cluster_labels[i]].append(i)
            else:
                cluster_index_groups[cluster_labels[i]] = [i]

        df_list = []
        for cluster_label in cluster_index_groups:
            cluster_image_indices = cluster_index_groups[cluster_label]
            cluster_image_names = self.image_names[cluster_image_indices]
            df_list.extend(list(zip([cluster_label for _ in range(0, len(cluster_image_names))], cluster_image_names)))
        df = pd.DataFrame(df_list)
        if self.cluster_type == "KMEANS" or self.cluster_type == "KMEDOIDS":
            csv_save_name = f"../files/{self.cluster_type}_k={num_clusters}_{Path(self.embedding_path).stem}.csv"
        if self.cluster_type == "DBSCAN":
            csv_save_name = f"../files/{self.cluster_type}_eps={eps:.2f}_{Path(self.embedding_path).stem}.csv"
        df.to_csv(csv_save_name)

        if self.cluster_type == "KMEDOIDS":
            center_save_name = f'../files/{self.cluster_type}_centerIDs_k={num_clusters}_{Path(self.embedding_path).stem}'
            np.save(center_save_name, center_ids)
        return csv_save_name
    
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
    ClusterGen = Clusterer(cluster_type="KMEANS", embedding_path='../embeddings/tmdb_both_5000.pkl')
    ClusterGen.save_clusters(num_clusters=20)


           
                

            





    


    
    

    


     

    