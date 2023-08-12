import numpy as np 
import pandas as pd 
from datasets import load_dataset, load_from_disk 
import cv2
import matplotlib.pyplot as plt 
import math 

class Cluster: 
    def __init__(self, image_idxs):
        self.image_idxs = image_idxs #list of hugging face image index (i.e profile is hf_dataset[i])
        self.numLikes = 0
        self.numDislikes = 0
        self.likedIdxs = [] #hugging face dataset image index
        self.dislikedIdxs = [] #hugging face dataset image index
    def getUCBScore(self, numRounds):
        avgRating = self.numLikes / (self.numDislikes+self.numLikes)
        numVisits = self.numLikes + self.numDislikes
        return avgRating + np.sqrt(np.log(numRounds)/numVisits)
    
    def get_random_image_idx(self):
        #returns un-seen hugging face image index i in cluster (i.e profile is hf_dataset[i])
        seen_images = set()
        seen_images.update(self.likedIdxs)
        seen_images.update(self.dislikedIdxs)
        ret_image = None
        while(ret_image is None):
            random_idx = np.random.randint(0, len(self.image_idxs),1)[0]
            if random_idx in seen_images: continue 
            else:
                ret_image = self.image_idxs[random_idx]
        return ret_image
    
    def update_values(self, rating, image_idx):
        if rating==1:
            self.numLikes += 1
            self.likedIdxs.append(image_idx)
        else:
            self.numDislikes += 1
            self.dislikedIdxs.append(image_idx)

    def print_stats(self):
        print(f'Number of likes {self.numLikes}')
        print(f'Number of dislikes {self.numDislikes}')
    

class OnlineEngine:
    '''
    Note: all indices in the clusters represent the hugging face indices, i.e index i corresponds to hf_dataset[i]
    '''
    def __init__(self, likesThreshold, clusters_file, sim_matrix_file, dataset_idxs_file, dataset_dir="../data/tmdb-people-image", numRounds=30):
        self.likesThreshold = likesThreshold
        self.clusters_file = clusters_file
        self.sim_matrix_file = sim_matrix_file
        self.clusters = []
        self.numClusters = None 
        self.hf_dataset_image_idxs = np.load(dataset_idxs_file).tolist() #arr[i] means that the i idxed vector in embedding matrix has hf_dataset index arr[i]
        self.generate_clusters()
        self.numRounds = numRounds 
        self.curRound = 1
        self.dataset = load_from_disk(dataset_dir)
        self.totalLikes = 0

    def generate_clusters(self):
        '''
        Cluster indices are the hugging face indices- i.e idx i means that we take hf_dataset[i]
        '''
        clusters_df = pd.read_csv(self.clusters_file, index_col=0)
        cluster_labels = clusters_df.iloc[:,0]
        image_names = clusters_df.iloc[:,1]
        hugging_indices = [int(name.split("_")[-1]) for name in image_names]
        cluster_index_groups = {} #key is cluster number, values is hugging face dataset indices (i.e hf_dataset[i])
        for i in range(0, len(cluster_labels)):
            if cluster_labels[i] in cluster_index_groups:
                cluster_index_groups[cluster_labels[i]].append(hugging_indices[i])
            else:
                cluster_index_groups[cluster_labels[i]] = [hugging_indices[i]]
        self.clusters = [Cluster(image_idxs=idxs) for cluster_key, idxs in cluster_index_groups.items()]
        self.numClusters = len(self.clusters)

    def draw_arm(self):
        #returns an image index to display and cluster index
        if self.curRound <= self.numClusters:
            #draw from cluster i
            cluster = self.clusters[self.curRound - 1]
            image_idx = cluster.get_random_image_idx()
            return image_idx, self.curRound - 1
        else:
            for i in range(0, len(self.clusters)):
                cluster = self.clusters[i]
                #print(f'Cluster {i}, round {self.curRound}  \n')
                cluster.print_stats()
                #print(f'UCB score is {cluster.getUCBScore(self.curRound)}')

            cluster_ucb_values = [cluster.getUCBScore(self.curRound) for cluster in self.clusters]
            max_idx = np.argmax(cluster_ucb_values)
            return self.clusters[max_idx].get_random_image_idx(), max_idx

    def round_driver(self):
        print("Round called")
        #display image 
        dataset_img_idx, cluster_idx = self.draw_arm()
        #print(f'Image index {dataset_img_idx} and cluster index {cluster_idx}')
        #print(f'Image index type {type(dataset_img_idx)}')
        profile = self.dataset[dataset_img_idx]
        cv2.imshow("Current Profile Picture", np.array(profile['image'])[...,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows() # destroy all windows
        user_response = input("Type y or n: \n")
        rating = (user_response == "y")
        if rating: self.totalLikes += 1
        self.clusters[cluster_idx].update_values(rating, dataset_img_idx)
        self.curRound += 1
        #print(f"Current round {self.curRound}")

    def game_run(self):
        print("Welcome to Find Your Type")
        #for _ in range(self.numRounds):
        #    self.round_driver()
        while (self.totalLikes < self.likesThreshold or self.curRound <= self.numRounds):
            self.round_driver()
        print("Finished rounds. Returning type...")
        type_image_indices = self.generate_type()
        liked_idxs, disliked_idxs = self.ret_liked_disliked_idxs()
        self.display_image(type_image_indices, "Recommended Category")
        self.display_image(liked_idxs, "Liked Images", num_rows=1)
        self.display_image(disliked_idxs, "Disliked Images", num_rows=3)



    def ret_liked_disliked_idxs(self):
        liked_idxs = [] #idxs correspond to hf_datset indices, i.e profile = hf_dataset[i]
        disliked_idxs = []
        for cluster in self.clusters:
            liked_idxs.extend(cluster.likedIdxs)
            disliked_idxs.extend(cluster.dislikedIdxs)
        return liked_idxs, disliked_idxs
 
    def generate_type(self, cluster_size=6):
        liked_idxs, disliked_idxs = self.ret_liked_disliked_idxs()
        #called after using inputs ratings
        imageGraph = ImageGraph(liked_idxs, disliked_idxs, self.hf_dataset_image_idxs, self.sim_matrix_file)
        SIM_THRESHOLD = 0.75
        #top_cluster is set of hugging face dataset indices 
        top_cluster = imageGraph.get_top_rated_cluster(cluster_size, SIM_THRESHOLD)
        return top_cluster 
    
    def display_image(self, dataset_indices, title = "", num_rows=2):
        cols = math.ceil(len(dataset_indices)/num_rows)
        fig = plt.figure(figsize=(num_rows * 5, cols*3))
        if title: fig.suptitle(title)
        for i in range(0, len(dataset_indices)):
            fig.add_subplot(num_rows, cols, i+1)
            profile = self.dataset[dataset_indices[i]]
            plt.imshow(np.array(profile['image']))
            plt.axis('off')
            #plt.title(f"Example {i+1}")
        plt.show()
        return



class ImageGraph:
    def __init__(self, likedIdxs, dislikedIdxs, hf_dataset_idxs, sim_matrix_file):
        self.likedIdxs = likedIdxs
        self.dislikedIdxs = dislikedIdxs
        self.hf_dataset_idxs = hf_dataset_idxs
        self.sim_matrix_file = sim_matrix_file
        self.sim_matrix = None 
        self.imageValues = [] #list of (hf_dataset_index, image_value)
        self.generateValues()

    def generateValues(self):
        '''
        populates imageValues list which contains tuples of (image_index, image_value)
        the value of each image is a weighted combination of values of rated points where weight is image similarity
        and the other liked images have +1 value and other disliked images have 0 value 
        NOTE: order of the image indices should match the order of the similarity matrix
        '''
        sim_matrix = np.load(self.sim_matrix_file)
        assert len(sim_matrix) == len(self.hf_dataset_idxs)
        self.sim_matrix = sim_matrix

        for i in range(0, len(self.hf_dataset_idxs)):
            dataset_image_idx = self.hf_dataset_idxs[i]
            weightedVal = 0
            for liked_image_idx in self.likedIdxs:
                embedding_idx = self.hf_dataset_idxs.index(liked_image_idx)
                similarity_val = sim_matrix[i][embedding_idx]
                weightedVal += similarity_val*1
            self.imageValues.append((dataset_image_idx, weightedVal))

    def get_top_rated_cluster(self, cluster_size, sim_threshold):
        '''
        Finds the maximal value point and greedily generates a cluster of points closest to it
        If it is not possible to generate a cluster around the maximal point such that the sim is larger than threshold, 
        it uses the second largest value point and so on  
        '''
        sorted_image_vals = sorted(self.imageValues, key=lambda x: x[1], reverse=True)
        #try to build cluster around highest point and find next best if not 
        for i in range(0, len(sorted_image_vals)):
            center_point = sorted_image_vals[i]
            center_emb_index = self.hf_dataset_idxs.index(center_point[0])
            cluster_embedding_indices = []
            #get most sim indices in descending order
            sorted_sim_point_indices = np.argsort(-1*self.sim_matrix[center_emb_index])
            assert sorted_sim_point_indices[0] == center_emb_index
            for j in range(0, cluster_size):
                other_idx = sorted_sim_point_indices[j]
                if self.sim_matrix[center_emb_index][other_idx] > sim_threshold:
                    cluster_embedding_indices.append(other_idx)
                else:
                    break 
            if len(cluster_embedding_indices) == cluster_size:
                cluster_dataset_idxs = [self.hf_dataset_idxs[embed_idx] for embed_idx in cluster_embedding_indices]
                return cluster_dataset_idxs
        return 
    
    
    def _get_top_rated_cluster(self, cluster_size, sim_threshold):
        '''
        NON-OPTIMIZED, NP hard, exponential
        Returns cluster (list of indices that corresponds to embedding matrix)
        '''
        lst_subset_vals = [] #contains tuples of (cluster_sum_val, cluster_indices)
        #find the max value subset of size cluster_size where each pairwise sim is larger than sim_threshold
        def backtrack(cur_lst, search_from_idx, num_items):
            #enumerate our imageValues tuples from [0, N), cur_lst contains the indices corresponding to the tuples
            if num_items == cluster_size:
                sum_val = 0
                for idx in cur_lst:
                    img_index, img_val = self.imageValues[idx]
                    sum_val += img_val 
                lst_subset_vals.append((sum_val, cur_lst))
                return 
            for i in range(search_from_idx, len(self.imageValues)):
                new_lst = cur_lst.copy()
                new_lst.append(i)
                if num_items > 0:
                    #check if new item has pairwise similarity larger than threshold for all in cur_lst
                    is_sim_enough = True 
                    for cur_elm_idx in cur_lst:
                        pairwise_sim_val = self.sim_matrix[i][cur_elm_idx]
                        if pairwise_sim_val < sim_threshold:
                            is_sim_enough = False 
                            break 
                    if not is_sim_enough: continue 
                backtrack(new_lst, i+1, num_items + 1)
            return
        backtrack([], 0, 0)
        max_cluster_obj = max(lst_subset_vals, key=lambda x: x[0])
        cluster_hf_dataset_indices = [] #hugging face dataset indices
        for idx in max_cluster_obj[1]:
            cluster_hf_dataset_indices.append(self.imageValues[idx][0])
        return cluster_hf_dataset_indices


    







if __name__ == "__main__":
    game = OnlineEngine(5, "../files/KMEANS_k=10_male_5000_5-5_nonan.csv", "../files/simMatrix_male_5000_5-5_nonan.npy",
                         "../files/hf_dataset_idxs_male_5000_5-5_nonan.npy", numRounds=30)
    game.game_run()
    
    
    








    






if __name__ == "__main__":
    pass 