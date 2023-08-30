import matplotlib.pyplot as plt
import numpy as np  
import os 
import pickle 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE


def generate_plot(user_rating_dict):
    '''
    user_rating_dict: (dict) key: tmdb id and val: rating (1 or 0)
    '''


def generate_plots_ratings(user_rating_dicts_file, tmdbid_embedding_file, tsne_embedding_matrix_file=None, perp=5):
    '''
    user_rating_dicts_file: (str) File path of a pickle of a list of dictionaries where key: tmdb id and val: rating (1 or 0)
    tmdbid_embedding_file: (str) File path of pickle of dataframe containing image name (w/ tmdb id) and embedding
    '''
    with open(user_rating_dicts_file, 'rb') as fp:
        user_rating_dicts = pickle.load(fp) 

    embedding_df = pd.read_pickle(tmdbid_embedding_file)
    image_names = embedding_df['image_names'] 
    tmdb_ids = [int(image_name.split("_")[-1]) for image_name in image_names]

    embeddings = np.stack(embedding_df['embeddings'].to_numpy())
    #scale data
    scaler = StandardScaler()
    scaled_embedding_matrix = scaler.fit_transform(embeddings)
    #PCA dimensionality reduction to 50
    pca = PCA(n_components=50)
    pca_embeddings = pca.fit_transform(scaled_embedding_matrix)
    tsne_embeddings = TSNE(n_components=2, verbose=1, perplexity=perp).fit_transform(pca_embeddings)

    n = len(user_rating_dicts)
    rows = int(np.ceil(np.sqrt(n))) 
    cols = int(np.ceil(n/ rows)) 

    fig,axs = plt.subplots(rows,cols, figsize=(18,12)) 
    axs = axs.flatten()

    for i in range(n):
        user_rating_dict = user_rating_dicts[i]
        labels  = np.array([int(user_rating_dict[id]) for id in tmdb_ids])
        liked_data = tsne_embeddings[labels == 1]
        disliked_data = tsne_embeddings[labels == 0]

        axs[i].scatter(x=disliked_data[:, 0], y=disliked_data[:, 1], color="blue", label="Disliked", alpha=0.25, s=10)
        axs[i].scatter(x=liked_data[:, 0], y=liked_data[:, 1], color="red", label="Liked", alpha=0.5, s=10)
        
    for i in range(n, rows * cols):
        axs[i].remove()
    plt.tight_layout()
    plt.show()

    return 









if __name__ == "__main__":
    generate_plots_ratings("./eval_files/3cluster_benchmark_2ratings_1cluster_female5000.pkl", "../embeddings/tmdb_female_5000.pkl")


