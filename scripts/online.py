import numpy as np 
import pandas as pd 
from datasets import load_dataset, load_from_disk 
import cv2

class Cluster: 
    def __init__(self, image_idxs):
        self.image_idxs = image_idxs
        self.numLikes = 0
        self.numDislikes = 0
        self.likedIdxs = []
        self.dislikedIdxs = []
    def getUCBScore(self, numRounds):
        avgRating = self.numLikes / (self.numDislikes+self.numLikes)
        numVisits = self.numLikes + self.numDislikes
        return avgRating + np.sqrt(np.log(numRounds)/numVisits)
    
    def get_random_image_idx(self):
        #returns un-seen image index in cluster
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
    

class OnlineEngine:
    def __init__(self, numRounds, clusters_file, dataset_dir="../data/tmdb-people-image"):
        self.clusters_file = clusters_file
        self.clusters = []
        self.numClusters = None 
        self.generate_clusters()
        self.numRounds = numRounds 
        self.curRound = 1
        self.dataset = load_from_disk(dataset_dir)

    def init_algo(self):
        self.onlineAlgo = UCBEnv(self.numRounds, self.clusters_file)

    def generate_clusters(self):
        clusters_df = pd.read_csv(self.clusters_file, index_col=0)
        cluster_labels = clusters_df.iloc[:,0]
        image_names = clusters_df.iloc[:,1]
        hugging_indices = [int(name.split("_")[-1]) for name in image_names]
        cluster_index_groups = {} #key is cluster number, values is hugging face indices
        for i in range(0, len(cluster_labels)):
            if cluster_labels[i] in cluster_index_groups:
                cluster_index_groups[cluster_labels[i]].append(hugging_indices[i])
            else:
                cluster_index_groups[cluster_labels[i]] = [hugging_indices[i]]
        self.clusters = [Cluster(image_idxs= idxs) for cluster_key, idxs in cluster_index_groups.items()]
        self.numClusters = len(self.clusters)

    def round_driver(self):
        #display image 
        display_img_idx, cluster_idx = self.onlineAlgo.draw_arm()
        print(f'Image index {display_img_idx} and cluster index {cluster_idx}')
        print(f'Image index type {type(display_img_idx)}')
        profile = self.dataset[display_img_idx]
        cv2.imshow("Current Profile Picture", np.array(profile['image'])[...,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows() # destroy all windows
        user_response = input("Type y or n: \n")
        rating = (user_response == "y")
        self.clusters[cluster_idx].update_values(rating, display_img_idx)
        self.curRound += 1
        self.onlineAlgo.increment_round()
        print(f"Current round {self.curRound}")

    def game_run(self):
        print("Welcome to Find Your Type")
        for _ in range(self.numRounds):
            self.round_driver()
        
        for cluster in self.clusters:
            print(f"Liked: {cluster.likedIdxs}")
            print(f'Disliked: {cluster.dislikedIdxs}')

class UCBEnv(OnlineEngine):
    def __init__(self, numRounds, clusters_file):
        super().__init__(numRounds, clusters_file)

    def draw_arm(self):
        #returns an image index to display and cluster index
        print(f"UCB current round {self.curRound}")
        if self.curRound <= self.numClusters:
            #draw from cluster i
            cluster = self.clusters[self.curRound - 1]
            image_idx = cluster.get_random_image_idx()
            return image_idx, self.curRound - 1
        else:
            cluster_ucb_values = [cluster.getUCBScore(self.curRound) for cluster in self.clusters]
            max_idx = np.argmax(cluster_ucb_values)
            return self.clusters[max_idx].get_random_image_idx(), max_idx
    def increment_round(self):
        self.curRound += 1
        
        

if __name__ == "__main__":
    game = OnlineEngine(30, "../files/KMEANS_k=100_female_5000_5-5_nonan.csv")
    game.init_algo()
    game.game_run()
    
    
    








    






if __name__ == "__main__":
    pass 