from unittest.mock import Mock
import main


def test_rec_ids():
    bucket_name = "similarity-matix"
    sim_file = 'simMatrix_both_5000_nonan.npy'
    sim_id_file = 'tmdb_ids_both_5000_nonan.npy'
    liked_ids = [6614,1907997,1136406]
    disliked_ids = [33260,117642,18918,111924,56680]
    sim_threshold = 0.75
    cluster_size = 6 
    data = {"bucket": bucket_name, "sim_file": sim_file, "sim_id_file": sim_id_file,
            "liked_ids": liked_ids, "disliked_ids": disliked_ids, "cluster_size": cluster_size, "sim_threshold": sim_threshold} 

    req = Mock(get_json=Mock(return_value=data), args=data) 
    response = main.matrix_read(req) 
    assert response == "True"
    print(response)
