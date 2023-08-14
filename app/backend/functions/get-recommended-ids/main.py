import functions_framework
from google.cloud import storage 
import numpy as np 
import io 


def generateValues(sim_matrix, sim_ids, liked_ids):
    '''
    Param: sim_ids: (1D array). The ith value is the tmdbID corresponding to ith row in sim_matrix
    Ret: list which contains tuples of (image_index, image_value)
    the value of each image is a weighted combination of values of rated points where weight is image similarity
    and the other liked images have +1 value and other disliked images have 0 value 
    '''
    imageValues = []
    for i in range(0, len(sim_ids)):
        weightedVal = 0
        for liked_id in liked_ids:
            embedding_idx = np.argwhere(sim_ids == liked_id)[0][0]
            similarity_val = sim_matrix[i][embedding_idx]
            weightedVal += similarity_val*1
        imageValues.append((int(sim_ids[i]), float(weightedVal)))
    return imageValues


def get_top_rated_cluster(imageValues, sim_matrix, sim_ids, cluster_size, sim_threshold):
    '''
    Finds the maximal value point and greedily generates a cluster of points closest to it
    If it is not possible to generate a cluster around the maximal point such that the sim is larger than threshold, 
    it uses the second largest value point and so on  
    '''
    sorted_image_vals = sorted(imageValues, key=lambda x: x[1], reverse=True)
    #try to build cluster around highest point and find next best if not 
    for i in range(0, len(sorted_image_vals)):
        center_point = sorted_image_vals[i]
        center_emb_index = np.argwhere(sim_ids == center_point[0])[0][0]
        cluster_embedding_indices = []
        #get most sim indices in descending order
        sorted_sim_point_indices = np.argsort(-1*sim_matrix[center_emb_index])
        assert sorted_sim_point_indices[0] == center_emb_index
        for j in range(0, cluster_size):
            other_idx = sorted_sim_point_indices[j]
            if sim_matrix[center_emb_index][other_idx] > sim_threshold:
                cluster_embedding_indices.append(int(other_idx))
            else:
                break 
        if len(cluster_embedding_indices) == cluster_size:
            cluster_ids = [int(sim_ids[embed_idx]) for embed_idx in cluster_embedding_indices]
            return cluster_ids 
    return 


@functions_framework.http
def matrix_read(request):
    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}
    request_json = request.get_json() 
    if 'bucket' not in request_json or 'sim_file' not in request_json or \
    'sim_id_file' not in request_json or 'liked_ids' not in request_json or \
    'disliked_ids' not in request_json or 'cluster_size' not in request_json or 'sim_threshold' not in request_json:
        return ("Missing parameters", 400, headers)
    
    bucket_name = request_json['bucket']
    sim_matrix_file = request_json['sim_file']
    sim_id_file = request_json['sim_id_file']
    liked_ids = request_json['liked_ids']
    cluster_size = request_json['cluster_size']
    sim_threshold = request_json['sim_threshold']
    disliked_ids = request_json['disliked_ids']

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    sim_matrix_blob, sim_id_blob = bucket.blob(sim_matrix_file), bucket.blob(sim_id_file)
    
    sim_matrix_bytes, sim_id_bytes = sim_matrix_blob.download_as_bytes(), sim_id_blob.download_as_bytes()
    sim_ids = np.load(io.BytesIO(sim_id_bytes))
    sim_matrix = np.load(io.BytesIO(sim_matrix_bytes))

    imageValues = generateValues(sim_matrix, sim_ids, liked_ids)
    rec_ids = get_top_rated_cluster(imageValues, sim_matrix, sim_ids, cluster_size, sim_threshold) 
    return(f"{','.join(map(str, rec_ids))}", 200, headers)





    