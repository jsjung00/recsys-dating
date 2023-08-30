import numpy as np 
import pandas as pd 
import random 

class AlgoBase:
    def __init__(self):
        self.curRound = 1
         
    def reset(self):
        self.curRound = 1 
    
    def select_image(self):
        print("Should implement select_image method")

    def add_rating(self):
        print("Add rating")

    def update_round(self):
        self.curRound += 1


class Cluster: 
    def __init__(self, image_ids):
        self.image_ids = image_ids #list of tmdb ids
        self.numLikes = 0
        self.numDislikes = 0
        self.likedIds = [] #tmdb ids 
        self.dislikedIds = [] #tmdb ids 

    def getUCBScore(self, numRounds):
        avgRating = self.numLikes / (self.numDislikes+self.numLikes)
        numVisits = self.numLikes + self.numDislikes
        return avgRating + np.sqrt(np.log(numRounds)/numVisits)
    
    def get_random_image_id(self):
        #returns un-seen tmdb id in cluster
        seen_images = set()
        seen_images.update(self.likedIds)
        seen_images.update(self.dislikedIds)
        ret_image = None
        while(ret_image is None):
            random_idx = np.random.randint(0, len(self.image_ids))  
            random_id = self.image_ids[random_idx] 
            if random_id in seen_images: continue 
            else:
                return random_id 
    
    def update_values(self, rating, image_id):
        if rating==1:
            self.numLikes += 1
            self.likedIds.append(image_id)
        else:
            self.numDislikes += 1
            self.dislikedIds.append(image_id)

    def print_stats(self):
        print(f'Number of likes: {self.numLikes}. Number of dislikes: {self.numDislikes}')
    

class UCB(AlgoBase):
    def __init__(self, cluster_csv_file):
        super().__init__()
        self.cluster_csv_file = cluster_csv_file 
        self.lastDrawnCluster = None 
        self.clusters = []
        self.numClusters = None 
        self.generate_clusters()

    def reset(self):
        super().reset()
        self.lastDrawnCluster = None 
        self.generate_clusters() 

    def generate_clusters(self):
        clusters_df = pd.read_csv(self.cluster_csv_file, index_col=0)
        cluster_labels = clusters_df.iloc[:,0]
        image_names = clusters_df.iloc[:,1]
        tmdb_ids = [int(name.split("_")[-1]) for name in image_names]
        cluster_index_groups = {} #key is cluster number, values is list of tmdb_ids
        for i in range(0, len(cluster_labels)):
            if cluster_labels[i] in cluster_index_groups:
                cluster_index_groups[cluster_labels[i]].append(tmdb_ids[i])
            else:
                cluster_index_groups[cluster_labels[i]] = [tmdb_ids[i]]
        self.clusters = [Cluster(image_ids=ids) for cluster_key, ids in cluster_index_groups.items()]
        self.numClusters = len(self.clusters)

    def select_image(self):
        #returns an image_id
        if self.curRound <= self.numClusters:
            #draw from cluster i
            cluster = self.clusters[self.curRound - 1]
            image_id = cluster.get_random_image_id()
            self.lastDrawnCluster = cluster 
            return image_id
        else:
            cluster_ucb_values = [cluster.getUCBScore(self.curRound) for cluster in self.clusters]
            max_idx = np.argmax(cluster_ucb_values)
            self.lastDrawnCluster = self.clusters[max_idx]
            return self.clusters[max_idx].get_random_image_id()
    
    def add_rating(self, rating, image_id):
        self.lastDrawnCluster.update_values(rating, image_id)

    def print_cluster_ratings(self):
        for i in range(0, len(self.clusters)):
            print(f"Cluster {i}", end="")
            self.clusters[i].print_stats()
    
    def print_cluster_ucb_vals(self):
        if self.curRound > self.numClusters:
            for i in range(0, len(self.clusters)):
                print(f"Cluster {i}", end=": ")
                print(self.clusters[i].getUCBScore(self.curRound))

class MinSim(AlgoBase):
    '''
    Greedily chooses point x that maximizes objective function: val(x) - gamma*maxSimToAnyRatedPoint(x)
    '''
    def __init__(self, gamma, tmdbid_to_rowidx_file, sim_matrix_file):
        super().__init__()
        self.gamma = gamma 
        tmdbid_to_rowidx = np.load(tmdbid_to_rowidx_file).tolist()
        self.tmdbid_to_rowidx = tmdbid_to_rowidx
        self.likedIds = []
        self.dislikedIds = [] 
        self.imageGraph = ImageGraph([], [], tmdbid_to_rowidx, sim_matrix_file)

    def select_image(self):
        ''' returns image_id ''' 
        #Draw random image in first round 
        if self.curRound == 1:
            return np.random.choice(self.tmdbid_to_rowidx)
        
        #Greedily select image that maximizes objective function 
        maxSimArr = self.imageGraph.get_max_sim_to_visited()
        print(f"maxsimarr length {len(maxSimArr)}")
        imageValTups = self.imageGraph.imageValues 
        objectiveArr = [imageValTups[i][1] - (self.gamma*maxSimArr[i]) for i in range(0, len(maxSimArr))]
        max_idx = np.argmax(np.array(objectiveArr))
        return self.tmdbid_to_rowidx[max_idx]    
    
    def reset(self):
        super().reset()
        self.likedIds = []
        self.dislikedIds = []  

    def add_rating(self, rating, image_id):
        if rating == 1:
            self.likedIds.append(image_id)
        else:
            self.dislikedIds.append(image_id)
        #update our imagegraph based on new values 
        self.imageGraph.likedIds = self.likedIds 
        self.imageGraph.dislikedIds = self.dislikedIds 
        self.imageGraph.generateValues() 

class EpsGreedy(AlgoBase):
    '''
    Randomly sample from unrated items epsilon number of times, else randomly choose a liked point and return
    the point closest to it (if no liked point, randomly sample from unrated)
    '''
    def __init__(self, eps, tmdbid_to_rowidx_file, sim_matrix_file):
        super().__init__()
        tmdbid_to_rowidx = np.load(tmdbid_to_rowidx_file).tolist()
        sim_matrix = np.load(sim_matrix_file)
        self.tmdbid_to_rowidx = tmdbid_to_rowidx
        self.sim_matrix = sim_matrix 
        self.eps = eps 
        self.likedIds = []
        self.dislikedIds = []  

    def select_image(self):
        ratedIds = set(self.likedIds) | set(self.dislikedIds)
        #randomly select unrated image if no liked or epsilon chance
        if len(self.likedIds) == 0 or np.random.rand() < self.eps:
            while(True):
                randomID = np.random.choice(self.tmdbid_to_rowidx)
                if randomID not in ratedIds:
                    return randomID 
        else:
            #randomly choose liked image, return unseen image closest to it 
            randomLikedID = np.random.choice(self.likedIds) 
            randomLikedIdx = self.tmdbid_to_rowidx.index(randomLikedID)
            sim_vector = self.sim_matrix[randomLikedIdx]
            low_high_sim_idxs = np.argsort(sim_vector)
            top_indices = low_high_sim_idxs[-5:]

            for i in range(len(low_high_sim_idxs)-1,-1,-1):
                nextIdx = low_high_sim_idxs[i] 
                nextId = self.tmdbid_to_rowidx[nextIdx] 
                if nextId not in ratedIds:
                    return nextId 
    def reset(self):
        super().reset()
        self.likedIds = []
        self.dislikedIds = []  

    def add_rating(self, rating, image_id):
        if rating == 1:
            self.likedIds.append(image_id)
        else:
            self.dislikedIds.append(image_id)
    
    





class ImageGraph:
    def __init__(self, likedIds, dislikedIds, tmdbid_to_rowidx, sim_matrix_file):
        self.likedIds = likedIds
        self.dislikedIds = dislikedIds
        self.tmdbid_to_rowidx = tmdbid_to_rowidx
        self.sim_matrix_file = sim_matrix_file
        self.sim_matrix = None 
        self.imageValues = [] #list of (tmdb_id, image_value)
        self.generateValues()

    def generateValues(self):
        '''
        Populates imageValues which contains tuples of (tmdb_id, image_value).
        imageValues[i] represents the tmdb_id and value of the ith row 
        The value is a weighted combination of the rated images values where weight is cosine similarity
        '''
        LIKED_VAL = 1
        DISLIKED_VAL = -0.25
        if len(self.likedIds) == 0 and len(self.dislikedIds) == 0:
            return 
        
        sim_matrix = np.load(self.sim_matrix_file)
        assert len(sim_matrix) == len(self.tmdbid_to_rowidx)
        self.sim_matrix = sim_matrix

        for i in range(0, len(self.tmdbid_to_rowidx)):
            tmdb_id = self.tmdbid_to_rowidx[i]
            weightedVal = 0
            for liked_image_id in self.likedIds:
                embedding_idx = self.tmdbid_to_rowidx.index(liked_image_id)
                similarity_val = sim_matrix[i][embedding_idx]
                weightedVal += similarity_val*LIKED_VAL
           
            for disliked_image_id in self.dislikedIds:
                embedding_idx = self.tmdbid_to_rowidx.index(disliked_image_id)
                similarity_val = sim_matrix[i][embedding_idx]
                weightedVal += similarity_val*(DISLIKED_VAL)
            self.imageValues.append((tmdb_id, weightedVal))
    
    def get_top_rated_id(self):
        #returns the id with the max value 
        sorted_image_vals = sorted(self.imageValues, key=lambda x: x[1], reverse=True)
        return sorted_image_vals[0][0] 

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
            center_emb_index = self.tmdbid_to_rowidx.index(center_point[0])
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
                cluster_ids = [self.tmdbid_to_rowidx[embed_idx] for embed_idx in cluster_embedding_indices]
                return cluster_ids 
        return 

    def get_max_sim_to_visited(self):
        '''
        Returns an array where A[i] represents the max similarity of row i to any
        rated vector  
        '''
        maxSimArr = [] 
        rated_ids = [*self.likedIds, *self.dislikedIds]
        rated_idxs = [self.tmdbid_to_rowidx.index(id) for id in rated_ids]
        for i in range(0, len(self.tmdbid_to_rowidx)):
            maxSimToRated = max([self.sim_matrix[i][rated_idx] for rated_idx in rated_idxs])
            maxSimArr.append(maxSimToRated)
        return maxSimArr           


    
    
