import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
import sys 
sys.path.append("../")
from scripts.clustering import Clusterer

def gen_cluster_synthetic_users(num_users, save_location, partition_size=30, num_clusters_dist = [1], no_duplicate_cluster=True):
    '''
    Partitions data into clusters, randomly chooses X number of clusters and makes all those points liked

    num_users: (int) Number of users to generate data for
    save_location: (str) Location to save array of rating dicts
    partition_size: (int) Number of clusters to parition the dataspace into 
    num_clusters_dist: (list) A[i] is the prob that user will have i+1 clusters  
    no_duplicate_cluster: (bool) If true, prevents different users from having the same liked cluster 
    Returns: list of dictionaries where key is tmdb_id, value is rating (1 or 0)
    '''
    Cluster = Clusterer()
    cluster_df = Cluster.ret_clusters_dict(partition_size)
    cluster_keys = set(list(cluster_df.keys())) 
    liked_clusters = set()

    user_ratings = []
    #each user has some number of clusters inside [1, len(num_clusters_dist)] 
    user_num_clusters = np.random.choice(np.arange(1, len(num_clusters_dist)+1), size=num_users, p=num_clusters_dist)
    for i in range(0, num_users):
        user = {}
        num_clusters = user_num_clusters[i]
        chosen_clusters = np.random.choice(list(cluster_keys-liked_clusters), size=num_clusters, replace=False)
        for chosen_cluster in chosen_clusters:
            if no_duplicate_cluster: liked_clusters.add(chosen_cluster)
            liked_imgs = cluster_df[chosen_cluster]
            for id in liked_imgs:
                user[id] = 1

        for non_chosen_cluster in list(cluster_keys-set(chosen_clusters)):
            disliked_imgs = cluster_df[non_chosen_cluster] 
            for id in disliked_imgs:
                user[id] = 0
        user_ratings.append(user) 
    
    if save_location is not None:
        with open(save_location, "wb") as fp:   #Save user ratings 
            pickle.dump(user_ratings, fp)
    
    return user_ratings

def gen_stochastic_synthetic_user(num_clusters, sim_matrix, tmdbid_to_rowidx_arr):
    '''
    num_clusters: (int) represents the number of types the user should have
    sim_matrix: (2d numpy array) cosine sim matrix
    tmdbid_to_rowidx_arr: (array or list) where A[i] is the tmdbID of the ith row in embedding matrix
    Returns: dictionary where key is tmdb_id, value is rating (1 or 0)
    '''
    num_people = len(sim_matrix)
    chosen_clusters_idx = np.random.choice(num_people, size=num_clusters, replace=False).tolist()
    print("clusters list", chosen_clusters_idx)
    id_rating_dict = {} #key is tmdb_id, value is rating (1 or 0)
    #assign the rating for each point, given by the piecewise linear function f(x)
    #where x is the cosine similiary to the closest cluster 
    vals = sim_matrix.flatten()
    median_sim = np.percentile(vals, 50) #1% chance like
    uq_sim =  np.percentile(vals, 75) #10% chance like
    ninety_sim = np.percentile(vals,90) #50% chance like 
    def get_like_prob(sim_val):
        med_prob = 0.01
        uq_prob = 0.1 
        ninety_prob = 0.5 
        if sim_val < median_sim:
            return sim_val/median_sim * med_prob
        elif sim_val < uq_sim:
            return ((sim_val-median_sim)/(uq_sim-median_sim) * (uq_prob-med_prob)) + med_prob 
        elif sim_val < ninety_sim:
            return ((sim_val-uq_sim)/(ninety_sim - uq_sim) * (ninety_prob - uq_prob)) + uq_prob
        else: 
            return ((sim_val-ninety_sim)/(1-ninety_sim) * (1-ninety_prob)) + ninety_prob



    for point_idx in range(0, num_people):
        tmdb_id = tmdbid_to_rowidx_arr[point_idx]
        if point_idx in chosen_clusters_idx:
            id_rating_dict[tmdb_id] = 1
        else:
            sim_vals = [sim_matrix[point_idx][center_idx] for center_idx in chosen_clusters_idx]
            max_sim = max(sim_vals)
            like_prob = get_like_prob(max_sim)
            rating = np.random.binomial(n=1, p=like_prob)
            id_rating_dict[tmdb_id] = rating 
    return id_rating_dict 

def gen_stochastic_synthetic_users(num_users, sim_matrix_file, tmdbid_to_rowidx_file, save_location, num_clusters_dist = [1]):
    '''
    num_users: (int) Number of users to generate data for
    save_location: (str) Location to save array of rating dicts
    num_clusters_dist: (list) A[i] is the prob that user will have i+1 clusters  
    '''
    tmdbid_to_rowidx_arr = np.load(tmdbid_to_rowidx_file)
    sim_matrix = np.load(sim_matrix_file)
    user_ratings = []
    #each user has some number of clusters inside [1, len(num_clusters_dist)] 
    user_num_clusters = np.random.choice(np.arange(1, len(num_clusters_dist)+1), size=num_users, p=num_clusters_dist)
    for i in range(0, num_users):
        num_clusters = user_num_clusters[i]
        new_user = gen_stochastic_synthetic_user(num_clusters,sim_matrix, tmdbid_to_rowidx_arr)
        user_ratings.append(new_user)
    
    if save_location is not None:
        with open(save_location, "wb") as fp:   #Save user ratings 
            pickle.dump(user_ratings, fp)
    
    return user_ratings


if __name__ == "__main__":
    #gen_cluster_synthetic_users(100, save_location= "./eval_files/10cluster_benchmark_100ratings_1cluster_female5000.pkl", partition_size=10, num_clusters_dist = [1], no_duplicate_cluster=False)
    gen_stochastic_synthetic_users(100, "../files/simMatrix_female_5000_5-5_nonan.npy", "../files/tmdbid_to_rowidx_tmdb_female_5000.npy",\
                                    save_location= "./eval_files/synthetic_benchmark_100ratings_1cluster_female5000.pkl", num_clusters_dist = [1])
    '''
    sim_matrix = np.load("../files/simMatrix_female_5000_5-5_nonan.npy")
    vals = sim_matrix.flatten()
    print(f'Half {np.percentile(vals, 50)}')
    print(f'75 {np.percentile(vals, 75)}')
    print(f'25 {np.percentile(vals, 25)}')
    tmdbid_to_rowidx_arr = np.load("../files/tmdbid_to_rowidx_tmdb_female_5000.npy")
    user = gen_stochastic_synthetic_user(1,sim_matrix,tmdbid_to_rowidx_arr)
    ratings = np.array(list(user.values()))
    num_liked = len(ratings[ratings==1])
    num_disliked = len(ratings[ratings==0])
    print(f"Num liked {num_liked}")
    print(f"Num disliked {num_disliked}")

   
    gen_stochastic_synthetic_users(num_users=10, sim_matrix_file="../files/simMatrix_female_5000_5-5_nonan.npy", \
                             tmdbid_to_rowidx_file="../files/tmdbid_to_rowidx_tmdb_female_5000.npy", save_location= "./eval_files/benchmark_10_ratings_1_cluster_female5000.pkl", \
                                  num_clusters_dist=[1])
   '''


     



