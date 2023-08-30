import numpy as np 
import pandas as pd 
import pickle 
from datasets import load_dataset, load_from_disk 
import cv2
import matplotlib.pyplot as plt 
import math 
from diversity import UCB, ImageGraph 

'''
Python program to run app on CPU. 
'''
    

class OnlineEngine:
    def __init__(self, Algo, likesThreshold, clusters_file, sim_matrix_file, tmdbid_hfidx_map_file, tmdbid_to_row_file, dataset_dir="../data/tmdb-people-image", numRounds=30):
        self.Algo = Algo 
        self.likesThreshold = likesThreshold
        self.clusters_file = clusters_file
        self.sim_matrix_file = sim_matrix_file
        with open(tmdbid_hfidx_map_file, "rb") as file:
            #dictionary where key is tmdbID, value is hf_index, i.e profile is A[hf_index] 
            self.tmdbid_hfidx_map = pickle.load(file)
        self.tmdbid_to_row_file = tmdbid_to_row_file
        self.numRounds = numRounds 
        self.curRound = 1
        self.dataset = load_from_disk(dataset_dir)
        self.totalLikes = 0

    def round_driver(self):
        print("Round called")
        #display image 
        img_id = self.Algo.select_image()
        if self.curRound > 10:
            self.Algo.print_cluster_ucb_vals()

        hf_idx = self.tmdbid_hfidx_map[img_id] 
        profile = self.dataset[hf_idx]
        cv2.imshow("Current Profile Picture", np.array(profile['image'])[...,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        user_response = input("Type y or n: \n")
        rating = (user_response == "y")
        if rating: self.totalLikes += 1
        self.Algo.add_rating(rating, img_id)
        self.Algo.update_round()
        self.curRound += 1
        print(f"Current round {self.curRound}")

    def game_run(self):
        print("Welcome to Find Your Type")
        while (self.totalLikes < self.likesThreshold or self.curRound <= self.numRounds):
            self.round_driver()
        print("Finished rounds. Returning type...")
        type_image_ids = self.generate_type()
        liked_ids, disliked_ids = self.ret_liked_disliked_ids()
        self.display_image(type_image_ids, "Recommended Category")
        self.display_image(liked_ids, "Liked Images", num_rows=1)
        self.display_image(disliked_ids, "Disliked Images", num_rows=3)



    def ret_liked_disliked_ids(self):
        liked_ids = [] #idxs correspond to hf_datset indices, i.e profile = hf_dataset[i]
        disliked_ids = []
        for cluster in self.Algo.clusters:
            liked_ids.extend(cluster.likedIds)
            disliked_ids.extend(cluster.dislikedIds)
        return liked_ids, disliked_ids
 
    def generate_type(self, cluster_size=6):
        liked_ids, disliked_ids = self.ret_liked_disliked_ids()
        tmdbid_to_rowidx = np.load(self.tmdbid_to_row_file).tolist()
        #called after using inputs ratings
        imageGraph = ImageGraph(liked_ids, disliked_ids, tmdbid_to_rowidx, self.sim_matrix_file)
        SIM_THRESHOLD = 0.75
        #top_cluster is set of hugging face dataset indices 
        top_cluster = imageGraph.get_top_rated_cluster(cluster_size, SIM_THRESHOLD)
        return top_cluster 
    
    def display_image(self, tmdb_ids, title = "", num_rows=2):
        dataset_indices = [self.tmdbid_hfidx_map[id] for id in tmdb_ids]
        
        cols = math.ceil(len(tmdb_ids)/num_rows)
        fig = plt.figure(figsize=(num_rows * 5, cols*3))
        if title: fig.suptitle(title)
        for i in range(0, len(tmdb_ids)):
            fig.add_subplot(num_rows, cols, i+1)
            profile = self.dataset[dataset_indices[i]]
            plt.imshow(np.array(profile['image']))
            plt.axis('off')
        plt.show()
        return








if __name__ == "__main__":
    ucb = UCB("../files/KMEANS_k=10_tmdb_female_5000.csv")
    game = OnlineEngine(ucb, 2, "../files/KMEANS_k=10_tmdb_female_5000.csv", "../files/simMatrix_female_5000_5-5_nonan.npy",
                         "../eval/eval_files/tmdbid_hfidx_map.pkl", "../files/tmdbid_to_rowidx_tmdb_female_5000.npy", numRounds=10)
    game.game_run()
    
    
    








    






if __name__ == "__main__":
    pass 