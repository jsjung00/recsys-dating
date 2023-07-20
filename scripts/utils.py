import numpy as np 
import os 
import shutil
from pathlib import Path
import pandas as pd 
DATA_PATH = "../data/cfd"
FOLDER_SAVE_PATH = "../clusters"

def generate_cluster_folders(df, cluster_folder_path):
    '''
    Given pandas dataframe containing cluster label and filename, create subfolders that contain all corresponding images
    '''
    cluster_labels = df.iloc[:, 1].values
    image_names = df.iloc[:,2].values 
    cluster_indices = {}
    for i in range(0, len(cluster_labels)):
        if cluster_labels[i] in cluster_indices:
            cluster_indices[cluster_labels[i]].append(i)
        else:
            cluster_indices[cluster_labels[i]] = [i]
    for k,v in cluster_indices.items():
        new_cluster_path = os.path.join(cluster_folder_path, f"cluster_{k}")
        print("new path", new_cluster_path)
        if not os.path.exists(new_cluster_path): os.mkdir(new_cluster_path)
        #add all images in the cluster to the cluster folder
        for image_folder_index in v:
            image_folder = image_names[image_folder_index]
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
            new_path = os.path.join(cluster_folder_path, f"cluster_{k}", f'{Path(image_path).stem}.png')
            shutil.copy(image_path, new_path)
    return 

def driver_generate_cluster_folders(cluster_csv_path):
    df = pd.read_csv(cluster_csv_path)
    #create a folder to save the cluster subfolders
    new_folder_path = os.path.join(FOLDER_SAVE_PATH, f"{Path(cluster_csv_path).stem}")
    if not os.path.exists(new_folder_path): os.mkdir(new_folder_path)
    generate_cluster_folders(df, new_folder_path)


def make_cfd_no_folders():
    '''
    Create one folder that contains all images with no subfolders
    '''
    new_folder_path = "../data/cfd_images"
    if os.path.exists(new_folder_path): 
        shutil.rmtree(new_folder_path)    
    os.mkdir(new_folder_path)

    i = 0
    for folder in os.listdir(DATA_PATH):
        images = os.listdir(os.path.join(DATA_PATH, folder))
        image_path = None   
        if len(images) > 1:
            for image_file in images:
                if Path(image_file).stem[-1] == "N":
                    image_path = os.path.join(DATA_PATH, folder, image_file) 
            if image_path is None:
                print(f"Folder {folder} contains no neutral image")
        elif len(images) == 0:
            print(f"Folder {folder} contains no images")
        else:
            image_path = os.path.join(DATA_PATH, folder, images[0])
        
        if image_path is not None:
            new_path = os.path.join(new_folder_path, f'{i}_{Path(image_path).stem}.png')
            shutil.copy(image_path, new_path)
        i += 1
    return 



if __name__ == "__main__":
    make_cfd_no_folders()
    #driver_generate_cluster_folders("../files/KMEANS_k=100_facenet512_CFD_embeddings.csv")
        
            



