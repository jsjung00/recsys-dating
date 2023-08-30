from datasets import load_dataset, load_from_disk 
import numpy as np 
import pandas as pd 
import pickle 

def relabel_embedding_to_tmdb():
    dataset = load_from_disk("../data/tmdb-people-image")
    embedding_df = pd.read_pickle("../embeddings/both_5000_5-5_nonan.pkl")
    image_names = embedding_df['image_names'].to_numpy()
    new_image_names = []
    for image_name in image_names:
        hf_idx = int(image_name.split("_")[-1])
        profile = dataset[hf_idx]
        tmdb_id = profile['id']
        new_name = f'{"_".join(image_name.split("_")[:-1])}_{tmdb_id}'
        new_image_names.append(new_name)
    
    embedding_df['image_names'] = new_image_names 
    print(embedding_df.head())

    embedding_df.to_pickle("../embeddings/tmdb_both_5000.pkl") 
    return 

def save_tmdb_id_to_hf_idx():
    dataset = load_from_disk("../data/tmdb-people-image")
    tmdbid_hfidx_map = {}
    for i in range(0, 30000):
        profile = dataset[i]
        tmdb_id = profile['id']
        tmdbid_hfidx_map[tmdb_id] = i 
    with open("./eval_files/tmdbid_hfidx_map.pkl", "wb") as file:
        pickle.dump(tmdbid_hfidx_map, file)
    
    return 

def get_user_avg_liked_sim(sim_matrix, user_rating_dict, tmdbid_to_idx_arr):
    tmdbid_to_idx = list(tmdbid_to_idx_arr)
    labels = np.array(list(user_rating_dict.values())) 
    liked_ids = np.array(list(user_rating_dict.keys()))[labels == 1] 
    liked_idxs = [tmdbid_to_idx.index(id) for id in liked_ids]
    total_sim_val = 0
    total_count = 0
    for i in range(0, len(liked_idxs) -1):
        sim_vals = sim_matrix[liked_idxs[i]][liked_idxs[i+1:]]
        total_sim_val += sum(sim_vals)
        total_count += len(sim_vals) 
    return total_sim_val/total_count 

def get_avg_liked_sim(sim_matrix_file,  user_ratings_file, tmdbid_to_idx_file):
    tmdbid_to_idx_arr = np.load(tmdbid_to_idx_file)
    sim_matrix = np.load(sim_matrix_file) 
    with open(user_ratings_file, 'rb') as fp:
        user_rating_dicts = pickle.load(fp) 
    for user_rating in user_rating_dicts:
        user_avg_liked_sim = get_user_avg_liked_sim(sim_matrix, user_rating, tmdbid_to_idx_arr)
        print(f"User has avg liked sim value of {user_avg_liked_sim}")
    return 

def print_sim_percentile_stats(sim_matrix_file):
    sim_matrix = np.load(sim_matrix_file)
    vals = sim_matrix.flatten()
    print(f'99 {np.percentile(vals, 99)}')
    print(f'90 {np.percentile(vals, 90)}')
    print(f'Half {np.percentile(vals, 50)}')
    print(f'75 {np.percentile(vals, 75)}')
    print(f'25 {np.percentile(vals, 25)}')

def print_num_likes(user_ratings_file):
    with open(user_ratings_file, 'rb') as fp:
        user_ratings = pickle.load(fp) 
    for i in range(0, len(user_ratings)):
        user_rating = user_ratings[i]
        ratings = np.array(list(user_rating.values()))
        num_likes = len(np.flatnonzero(ratings))
        num_dislikes = len(ratings) - num_likes 
        print(f"User {i} has {num_likes} likes and {num_dislikes} dislikes")

    
         


    

if __name__ == "__main__":
    print_num_likes("./eval_files/synthetic_benchmark_100ratings_1cluster_female5000.pkl")
    get_avg_liked_sim("../files/simMatrix_female_5000_5-5_nonan.npy","./eval_files/synthetic_benchmark_100ratings_1cluster_female5000.pkl"\
                      , "../files/tmdbid_to_rowidx_tmdb_female_5000.npy")
    
    with open("./eval_files/30cluster_benchmark_10ratings_1cluster_female5000.pkl", 'rb') as fp:
            user_rating_dicts = pickle.load(fp) 
    sim_matrix = np.load("../files/simMatrix_female_5000_5-5_nonan.npy")
    print_sim_percentile_stats("../files/simMatrix_female_5000_5-5_nonan.npy")
    #avg_liked_sim = get_user_avg_liked_sim(sim_matrix, user_rating_dicts[2])
    #print(f"Avg liked sim {avg_liked_sim}")

    #print_sim_percentile_stats("../files/simMatrix_female_5000_5-5_nonan.npy")
    #get_avg_liked_sim("../files/simMatrix_female_5000_5-5_nonan.npy", "./eval_files/benchmark_10_ratings_1_cluster_female5000.pkl")
