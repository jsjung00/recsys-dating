from deepface import DeepFace
import os 
import pandas as pd 
from pathlib import Path
from tqdm import tqdm

class CFDEmbeddingGenerator:
    def __init__(self, model="Facenet512"):
        self.model = model  

    def get_embedding(self, file_path):
        embedding_objs = DeepFace.represent(img_path = file_path, model_name = self.model)
        embedding = embedding_objs[0]["embedding"]
        return embedding 
    
    def get_embedding_table(self, data_dir="../data/cfd"):
        '''
        Open all images in dataset, save the file name and convert 
        image to vector embedding
        '''
        image_folders = os.listdir(data_dir)
        image_names = []
        image_embeddings = []

        for folder in tqdm(image_folders):
            images = os.listdir(os.path.join(data_dir, folder))
            image_path = None 
            if len(images) > 1:
                for image_file in images:
                    if Path(image_file).stem[-1] == "N":
                        image_path = os.path.join(data_dir, folder, image_file) 
                if image_path is None:
                    print(f"Folder {folder} contains no neutral image")
            elif len(images) == 0:
                print(f"Folder {folder} contains no images")
            else:
                image_path = os.path.join(data_dir, folder, images[0])

            if image_path is not None:
                image_names.append(folder)
                #convert image to embedding
                embedding = self.get_embedding(image_path)
                image_embeddings.append(embedding)
        
        #create a pandas dataframe that contains image name and embedding
        df = pd.DataFrame({"imageName": image_names, "imageEmbedding":image_embeddings})
        return df 
    
    


if __name__ == "__main__":
    CFDGen = CFDEmbeddingGenerator()
    embedding_table = CFDGen.get_embedding_table()
    embedding_table.to_csv("../embeddings/CFD_embeddings.csv")


