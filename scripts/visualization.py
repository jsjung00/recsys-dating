import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from pathlib import Path
import seaborn as sns 
import matplotlib.pyplot as plt 
import os 

class Visualizer():
    def __init__(self, embedding_file, id_idx_map_file, liked_ids, disliked_ids):
        self.embedding_file = embedding_file
        self.id_idx_map_file = id_idx_map_file
        self.liked_ids = liked_ids
        self.disliked_ids =  disliked_ids
    
    def return_idxs(self, ids_arr):
        #given an array of ids, returns list of corresponding row # in embedding matrix
        idxs_arr = np.load(self.id_idx_map_file)
        return [np.where(idxs_arr == id)[0][0] for id in ids_arr]
    
    def save_reduced_matrix(self):
        embedding_matrix = np.load(self.embedding_file)
        #scale data
        scaler = StandardScaler()
        scaled_embedding_matrix = scaler.fit_transform(embedding_matrix)
        #PCA dimensionality reduction to 50
        pca = PCA(n_components=50)
        pca_embeddings = pca.fit_transform(scaled_embedding_matrix)
        save_path = Path(self.embedding_file).with_name("pca50_" + Path(self.embedding_file).name) 
        np.save(save_path, pca_embeddings)
        print(pca_embeddings.shape)
        #tsne_embeddings = TSNE(n_components=2, perplexity=perplexity).fit_transform(pca_embeddings)

    def save_tsne_embedding(self, pca_file, perp):
        rest_name = "_".join(Path(pca_file).name.split("_")[1:])
        new_name = f"tsneperp={perp}_{rest_name}"
        save_path = Path(pca_file).with_name(new_name) 
        if os.path.exists(save_path):
             tsne_embeddings = np.load(save_path)
        else:
            pca_embeddings = np.load(pca_file)
            tsne_embeddings = TSNE(n_components=2, verbose=1, perplexity=perp).fit_transform(pca_embeddings)
            np.save(save_path, tsne_embeddings) 
        sns.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], color="blue")
        plt.savefig(f"../files/{new_name}.png")

    def save_across_perplexity(self, pca_file, perps):
         for perp in perps:
              self.save_tsne_embedding(pca_file, perp) 

    def create_plot(self, tsne_file):
        liked_idxs = self.return_idxs(self.liked_ids)
        disliked_idxs = self.return_idxs(self.disliked_ids) 

        tsne_embeddings = np.load(tsne_file)
        categories = [] #array of elm in (1,0,-1), where the value represents one of the 3 categories
        for i in range(0, len(tsne_embeddings)):
            if i in liked_idxs:
                categories.append("Liked")
            elif i in disliked_idxs:
                categories.append("Disliked")
            else:
                categories.append("No Rating") 

        # Set font and context
        sns.set(font="Arial")
        sns.set_context(context="notebook", font_scale=1.2)

        # Set style
        sns.set_style(style="whitegrid")         
        plot_df = pd.DataFrame({'x': tsne_embeddings[:, 0], 'y': tsne_embeddings[: ,1], 'category': categories})

        mypalette={"Liked": "#E57373", "Disliked": "#64B5F6", "No Rating": "#B0BEC5"}
        plt.figure(figsize=(6, 4.5), frameon=False)
        for cat in ["Disliked", "No Rating", "Liked"]:
            cat_data = plot_df[plot_df['category'] == cat]
            sns.scatterplot(data=cat_data, x='x', y='y', hue="category",palette=mypalette,
                size="category", sizes={"No Rating": 25, "Disliked": 50, "Liked": 50}, alpha=0.5 if cat == "No Rating" else 1.0, zorder=1 if cat == "No Rating" else 2)
            
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.legend(fontsize="10")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(f"../files/retPlot_{Path(tsne_file).name}.png", dpi=300)


    





















if __name__ == "__main__":
    disliked_ids = [
    68495,
    1685356,
    111513,
    25872,
    221809,
    1385063,
    1079533,
    129050,
    209038,
    38333,
    1465469,
    1244469,
    55776,
    238523,
    342100,
    1390968]

    liked_ids = [
    1253731,
    114869,
    24651,
    144813,
    1251426,
    3028798,
    1029719,
    68558,
    1492982,
    1015824,
    1252385,
    1136683,
    1288688,
    1122535]

    viz = Visualizer('..\embeddings\emb_matrix_female_5000_5-5_nonan.npy','../files/tmdb_ids_female_5000_nonan.npy', liked_ids, disliked_ids)
    #viz.save_across_perplexity('..\embeddings\pca50_emb_matrix_male_5000_5-5_nonan.npy', [5.2, 5.6, 6,7])
    viz.create_plot('../embeddings/tsneperp=5_emb_matrix_female_5000_5-5_nonan.npy')  
