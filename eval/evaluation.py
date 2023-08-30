import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
from tqdm import tqdm 
sys.path.append("../scripts")
from diversity import UCB, MinSim, EpsGreedy  
from clustering import Clusterer

class EvalEngine:
    def __init__(self, Algo, user_rating_dicts_file,tmdbid_to_rowidx_file, sim_matrix_file=None, numExplorationRounds=30):
        '''
        user_rating_dicts_file: (str) File path of a pickle of a list of dictionaries where key: tmdb id and val: rating (1 or 0)
        '''
        self.Algo = Algo 
        self.numExplorationRounds=numExplorationRounds
        self.curRound = 1
        with open(user_rating_dicts_file, 'rb') as fp:
            user_rating_dicts = pickle.load(fp) 
        self.user_ratings = user_rating_dicts
        self.rated_ids = [] 
        self.liked_ids = []
        self.sim_matrix_file = sim_matrix_file
        tmdbid_to_rowidx = np.load(tmdbid_to_rowidx_file).tolist()
        self.tmdbid_to_rowidx = tmdbid_to_rowidx 

    def reset(self):
        self.curRound = 1
        self.rated_ids = [] 
        self.liked_ids = []

    def round_driver(self, user_rating):
        '''
        Runs one exploration round.
        Returns: (int) 1 or 0 depending on like or dislike 
        '''
        img_id = self.Algo.select_image()
        self.rated_ids.append(img_id)
        img_rating = user_rating[int(img_id)]
        if img_rating == 1: self.liked_ids.append(img_id)
        self.Algo.add_rating(img_rating, img_id)
        self.Algo.update_round()
        self.curRound += 1
        return img_rating
        

    def run_simulation(self, user_rating):
        '''
        Runs one gameplay simulation. Returns array where A[i] represents number of likes on round i+1
        '''
        likesArr = [] 
        while (self.curRound <= self.numExplorationRounds):
            rating = self.round_driver(user_rating)
            likesArr.append(rating)       
        return likesArr
    def run_benchmark(self, showGraph=True, saveText=True, save_file_path="./eval_files/benchmark.txt", experiment_name="Experiment Name Here"):
        '''
        Returns array containing averaged (over users) likes by round i  
        '''    
        totalRunningLikesPerRoundArr = np.repeat(0, self.numExplorationRounds) 
        totalRunningLikesPerRoundArr = totalRunningLikesPerRoundArr.astype("float64")
        totalAvgPairSim = 0
        totalAvgLikedSim = 0 
        for user_rating in tqdm(self.user_ratings):
            self.Algo.reset() 
            self.reset()
            likesArr = self.run_simulation(user_rating)   
            totalLikesArr = np.cumsum(likesArr).tolist() #A[i] number of likes so far in round (i+1) 
            totalRunningLikesPerRoundArr += totalLikesArr
            #get average pairwise simiarity for all rated and all liked images for that user
            totalAvgPairSim += self.get_avg_pair_sim(self.rated_ids)
            if len(self.liked_ids) > 1: 
                totalAvgLikedSim += self.get_avg_pair_sim(self.liked_ids) 
        
        avgRunningLikesPerRoundArr = totalRunningLikesPerRoundArr/len(self.user_ratings) 
        avgPairSim = totalAvgPairSim/len(self.user_ratings)
        avgLikedSim = totalAvgLikedSim/len(self.user_ratings) 
        if showGraph:
            self.generate_plot(avgRunningLikesPerRoundArr)
        if saveText:
            self.save_stats_txt(avgRunningLikesPerRoundArr,avgPairSim,avgLikedSim, save_file_path, experiment_name)
        return 

    def save_stats_txt(self, arr, avg_rating_sim, avg_liked_sim, text_file_path, experiment_name):
        '''
        Save arr numbers and avg rating similarity in text file
        '''
        str_arr = [str(elm) for elm in arr]
        with open(text_file_path, 'a') as file:
            file.write(f"\n\n -----{experiment_name} ----- \n")
            file.write(f"Avg Running Likes:  [{','.join(str_arr)}] \n")
            file.write(f"Avg rating similarity: {str(avg_rating_sim)} \n")
            file.write(f"Avg liked similarity: {str(avg_liked_sim)} \n")
        return          

    def generate_plot(self, arr):
        plt.plot(np.arange(1, len(arr) + 1), arr)
        plt.title('Average Cumulative Likes')
        plt.xlabel('Round Number')
        plt.ylabel('Average Cumulative Likes')
        plt.show() 

    def get_avg_pair_sim(self, arr_ids):
        '''
        Return average pairwise similarity of an array of imageIDs. Lower means more diverse exploration 
        '''
        if self.sim_matrix_file is None: return 
        sim_matrix = np.load(self.sim_matrix_file) 
        total_sim_val = 0
        total_count = 0
        rated_idxs = [self.tmdbid_to_rowidx.index(id) for id in arr_ids]
        for i in range(0, len(rated_idxs)-1):
            sim_vals = sim_matrix[rated_idxs[i]][rated_idxs[i+1:]]
            total_sim_val += sum(sim_vals)
            total_count += len(sim_vals) 
        return total_sim_val/total_count 


class Experimenter:
    def __init__(self, param_arr, type, tmdbid_to_rowidx_file, sim_matrix_file, user_rating_dicts_file, showGraph=False, saveText=True):
        self.param_arr = param_arr
        self.type = type 
        self.tmdbid_to_rowidx_file = tmdbid_to_rowidx_file
        self.sim_matrix_file = sim_matrix_file 
        self.user_rating_dicts_file = user_rating_dicts_file 
        self.showGraph = showGraph
        self.saveText = saveText
    
    def ret_cluster_csv_file(self, k):
        '''
        k: (int) represents number of clusters 
        For UCB, return a cluster cvs file for k clusters 
        '''
        Clust = Clusterer()
        return Clust.save_clusters(num_clusters=k)

    def run_experiments(self, save_file_path="./eval_files/benchmark.txt"): 
        for param in self.param_arr:
            print(f"Starting experiment with parameter {param}")
            if self.type == "EPS":
                Algo = EpsGreedy(param,self.tmdbid_to_rowidx_file, self.sim_matrix_file)
            elif self.type == "UCB":
                Algo = UCB(self.ret_cluster_csv_file(param)) 
            else:
                print("Missing algo type. error")
                return 
            Eval = EvalEngine(Algo, self.user_rating_dicts_file, self.tmdbid_to_rowidx_file, self.sim_matrix_file) 
            Eval.run_benchmark(experiment_name=f"{self.type} param={param :.2f}",\
                                showGraph=self.showGraph, saveText=self.saveText, save_file_path=save_file_path) 




         

def test_run_simulation():
    Algo = EpsGreedy(0, "../files/tmdbid_to_rowidx_tmdb_female_5000.npy", "../files/simMatrix_female_5000_5-5_nonan.npy")
    Eval = EvalEngine(Algo, "./eval_files/3cluster_benchmark_2ratings_1cluster_female5000.pkl",\
                       "../files/tmdbid_to_rowidx_tmdb_female_5000.npy",\
                       "../files/simMatrix_female_5000_5-5_nonan.npy") 
    with open("./eval_files/3cluster_benchmark_2ratings_1cluster_female5000.pkl", 'rb') as fp:
            user_rating_dicts = pickle.load(fp) 
    likes_arr = Eval.run_simulation(user_rating_dicts[0]) 
    print(f"Expect {likes_arr}")

def test_benchmark():
    Algo = UCB("../files/KMEANS_k=10_tmdb_female_5000.csv")
    Eval = EvalEngine(Algo, "./eval_files/benchmark_10_ratings_1_cluster_female5000.pkl",\
                      "../files/tmdbid_to_rowidx_tmdb_female_5000.npy", "../files/simMatrix_female_5000_5-5_nonan.npy") 
    Eval.run_benchmark(saveText=True)

     




if __name__ == "__main__":
    Exp = Experimenter([0, 0.1, 0.2,0.3,0.5,0.9], "EPS", "../files/tmdbid_to_rowidx_tmdb_female_5000.npy", \
                            "../files/simMatrix_female_5000_5-5_nonan.npy", "./eval_files/10cluster_benchmark_100ratings_1cluster_female5000.pkl")
    Exp.run_experiments()
    
    '''
    run_epsilon_experiments([0, 0.2, 0.3, 1.0], "../files/tmdbid_to_rowidx_tmdb_female_5000.npy", \
                            "../files/simMatrix_female_5000_5-5_nonan.npy", "./eval_files/10cluster_benchmark_100ratings_1cluster_female5000.pkl")
    Algo = MinSim(0.5,"../files/tmdbid_to_rowidx_tmdb_female_5000.npy", "../files/simMatrix_female_5000_5-5_nonan.npy")
    Eval = EvalEngine(Algo, "./eval_files/benchmark_10_ratings_1_cluster_female5000.pkl"\
                      , "../files/tmdbid_to_rowidx_tmdb_female_5000.npy", "../files/simMatrix_female_5000_5-5_nonan.npy") 
    Eval.run_benchmark(saveText=True) 
    '''     

    
     

