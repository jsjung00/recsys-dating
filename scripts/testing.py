from get_embedding import GeneralEmbeddingGenerator
from affinity import return_similarity_vector
import os 
import pandas as pd 

def run_test_suite():
    if not os.path.exists("../embeddings/facenet512_test_embeddings.csv"):
        #generate the embeddings
        Gen = GeneralEmbeddingGenerator()
        embedding_table = Gen.get_embedding_table(data_dir="../data/test")
        embedding_table.to_csv("../embeddings/facenet512_test_embeddings.csv")
    
    sim_vec = return_similarity_vector(1, "../embeddings/facenet512_test_embeddings.csv")
    print(sim_vec)
    

if __name__ == "__main__":
    #run_test_suite()
  
