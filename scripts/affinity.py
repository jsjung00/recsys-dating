from datasets import load_dataset 
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np 
import pandas as pd 
import json 
import argparse
import os 
import shutil
from pathlib import Path

def return_similarity_vector(index, embeddings_file_path, metric="cosine"):
    '''
    index: (int)- represents the point of reference vector 
    Returns the similarity vector in the cosine similarity matrix
    '''
    embedding_data = pd.read_csv(embeddings_file_path)
    embeddings = embedding_data['imageEmbedding'].values
    image_names = embedding_data['imageName'].values 
    embeddings = np.array([json.loads(embedding) for embedding in embeddings])
    if metric=="cosine":
        similarity_matrix = cosine_similarity(embeddings)
    elif metric=="euclidean":
        similarity_matrix = euclidean_distances(embeddings)
    else:
        raise Exception("Only handles cosine metric for now.")
    return similarity_matrix[index]  


def return_k_closest(index, data, k, metric="cosine"):
    '''
    data: (2d array, represents list of feature vectors)
    Return k closest indices and their similarity values to a given index, where first element is closest (the given index itself)
    '''
    if metric=="cosine":
        similarity_matrix = cosine_similarity(data)
    elif metric=="euclidean":
        similarity_matrix = euclidean_distances(data)
    else:
        raise Exception("Only handles cosine metric for now.")
    print(f"Median cosine similarity: {np.median(similarity_matrix)}")
    print(f"Top 10% cosine similarity: {np.percentile(similarity_matrix, 90)}")
    
    similarity_vector = similarity_matrix[index]
    if metric=="cosine":
        #large values are more similar, so return k largest values
        top_k_indices = list(np.argsort(similarity_vector)[-k:]) 
        top_k_indices.reverse()
    elif metric=="euclidean":
        #small values are more similar, so return k smallest values
        top_k_indices = list(np.argsort(similarity_vector)[:k]) 

    return np.array(top_k_indices), similarity_vector[np.array(top_k_indices)]

SAVE_FOLDER = "../output"
DATA_PATH = "../data/cfd"
def generate_k_closest(index, embeddings_file_path, k, metric="cosine", data="hugging"):
    '''
    index: (int) represents the index of the desired point of ref in the data array
    Save the k closest images to a reference image in a folder
    '''
    if Path(embeddings_file_path).suffix == ".csv": 
        embedding_data = pd.read_csv(embeddings_file_path)
        embeddings = embedding_data['imageEmbedding'].values
        image_names = embedding_data['imageName'].values 
        embeddings = np.array([json.loads(embedding) for embedding in embeddings])  
    elif Path(embeddings_file_path).suffix == ".pkl":
        embedding_data = pd.read_pickle(embeddings_file_path)
        embeddings = np.stack(embedding_data['embeddings'].values)
        image_names = embedding_data['image_names'].values
    else:
        print(f'{embeddings_file_path} is not csv or pickle') 
    
    top_k_indices, sim_values = return_k_closest(index, embeddings, k, metric)
    new_folder = os.path.join(SAVE_FOLDER, "nearest_images")
    if os.path.exists(new_folder): 
        shutil.rmtree(new_folder)    
    os.mkdir(new_folder)

    if data == "hugging":
        hugging_indices = np.array([int(name.split("_")[-1]) for name in image_names])
        top_hugging_indices = hugging_indices[top_k_indices]
        celeb_faces = load_dataset("ashraq/tmdb-people-image", split='train')
        for i in range(0, len(top_hugging_indices)):
            idx = int(top_hugging_indices[i])
            profile = celeb_faces[idx]
            face = profile['image']
            if i == 0:
                new_path = os.path.join(new_folder, f"1original_ref_sim_{sim_values[i]:.2f}_Index_{idx}.png")
            else:
                new_path = os.path.join(new_folder, f"{i+1}_closest_ref_sim_{sim_values[i]:.2f}_Index_{idx}.png")
            face.save(new_path)
    else:
        #save the k closest in the folder in decreasing similarity order
        for i in range(0, len(top_k_indices)):
            image_folder = image_names[top_k_indices[i]]
            image_path = None
            image_files = os.listdir(os.path.join(DATA_PATH, image_folder)) 
            if len(image_files) > 1:
                for image_file in image_files:
                    if Path(image_file).stem[-1] == "N":
                        image_path = os.path.join(DATA_PATH, image_folder, image_file) 
                if image_path is None:
                    print(f"Folder {image_folder} contains no neutral image")
            elif len(image_files) == 0:
                print(f"Folder {image_folder} contains no images")
            else:
                image_path = os.path.join(DATA_PATH, image_folder, image_files[0])
            if i == 0:
                new_path = os.path.join(new_folder, f"1original_ref_sim_{sim_values[i]:.2f}_Index_{index}.png")
            else:
                new_path = os.path.join(new_folder, f"{i+1}_closest_ref_sim_{sim_values[i]:.2f}_Index_{index}.png")
            shutil.copy(image_path, new_path)
    return 

def similarity_between_indices(index1, indices, embeddings_file_path, metric="cosine"):
    vec = return_similarity_vector(index1, embeddings_file_path, metric)
    print(f"Similarity between index {index1} and indices {indices} is: {vec[np.array(indices)]}")
    return vec[np.array(indices)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",type=int, required=True, help="number of closest indices")
    parser.add_argument("--index", type=int, required=True, help="index to find the closest around")
    parser.add_argument("--indices", type=int, required=False, nargs="*", help="indices to get similarity with first")

    args = parser.parse_args()
    generate_k_closest(args.index,"../embeddings/female_5000_5-5_nonan.pkl",args.k, metric="cosine")
    #print("Finished saving k closest")
     
    