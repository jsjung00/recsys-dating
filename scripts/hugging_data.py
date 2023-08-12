#https://docs.pinecone.io/docs/facial-similarity-search
from pathos.multiprocessing import ProcessingPool as Pool
from datasets import load_dataset 
from datetime import datetime 
from deepface import DeepFace
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from multiprocessing import Pool
import math 

def get_valid_indices(array_size=5000):
    '''
    Returns two arrays of indices (male and female) which correspond to images of people that are 18-40
    '''
    female_indices = []
    male_indices = []
    celeb_faces = load_dataset("ashraq/tmdb-people-image", split='train')

    for i in range(100000):
        current_profile = celeb_faces[i]
        isDead = current_profile['deathday']
        if isDead is not None:
            continue 
        birthday_str = current_profile['birthday'] #YYYY-MM-DD
        if birthday_str is None: continue 
        birthday = datetime.strptime(birthday_str, '%Y-%m-%d')
        old_enough = datetime(2002, 7, 27)
        too_old = datetime(1970, 7,27)
        if birthday > old_enough or birthday < too_old:
            continue
        if current_profile['image'] is None or current_profile['name'] is None:
            continue 
        if current_profile['gender'] is None: continue 
        if current_profile['gender'] == 2 and len(male_indices) < array_size:
            male_indices.append(i)
        if current_profile['gender'] == 1 and len(female_indices) < array_size:
            female_indices.append(i) 
        else:
            if len(female_indices) >= array_size and len(male_indices) >= array_size:
                return female_indices, male_indices
        
        


def generate_embeddings(array_size=5000, get_female=False, get_male=True):
    def get_embedding(idx):
        profile = celeb_faces[idx]
        name, image = profile['name'], np.array(profile['image'])
        try:
            embedding_objs = DeepFace.represent(image, model_name="VGG-Face", detector_backend="mtcnn")
            embedding = embedding_objs[0]["embedding"]
        except:
            print(f"Image index {idx} failed to detect face")
            profile['image'].save(f"../files/failed_images/{name}_{idx}.jpg")
            embedding = None 
        image_name = f'{name}_{idx}'
        return (np.array(embedding), image_name)

    celeb_faces = load_dataset("ashraq/tmdb-people-image", split='train')
    female_embeddings, female_image_names = [], []
    male_embeddings, male_image_names = [], []
    female_idxs, male_idxs = get_valid_indices(array_size)

    if get_female:
        for i in range(0,5):
            partition_size = math.ceil(len(female_idxs)/5)
            idx_partition = female_idxs[i*partition_size: (i+1)*partition_size]
            partition_objects = map(get_embedding, tqdm(idx_partition))
            partition_embeddings, partition_names = list(zip(*partition_objects))
            female_embeddings.extend(partition_embeddings)
            female_image_names.extend(partition_names)
            female_embedding_df = pd.DataFrame({'image_names': female_image_names, 'embeddings':female_embeddings})
            female_embedding_df.to_pickle(f'../embeddings/female_{array_size}_{i+1}-5.pkl')

    if get_male:
        for i in range(4,5):
            partition_size = math.ceil(len(male_idxs)/5)
            idx_partition = male_idxs[i*partition_size: (i+1)*partition_size]
            partition_objects = map(get_embedding, tqdm(idx_partition))
            partition_embeddings, partition_names = list(zip(*partition_objects))
            male_embeddings.extend(partition_embeddings)
            male_image_names.extend(partition_names)
            male_embedding_df = pd.DataFrame({'image_names': male_image_names, 'embeddings': male_embeddings})
            male_embedding_df.to_pickle(f'../embeddings/male_{array_size}_{i+1}-5_only.pkl')

    return 
















if __name__ == "__main__":
    print("Starting script")
    #female_idxs, male_idxs = get_valid_indices(10)
    print(generate_embeddings(array_size=5000, get_male=True))