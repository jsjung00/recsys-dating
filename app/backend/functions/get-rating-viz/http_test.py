from unittest.mock import Mock
import main
import base64
from io import BytesIO
from PIL import Image
import json 

def test_send_viz():
    bucket_name = "embedding-files"
    id_idx_map_file = 'tmdb_ids_female_5000_nonan.npy'
    liked_ids = [1253731, 114869, 24651, 144813, 1251426, 3028798, 1029719, 68558, 1492982, 1015824, 1252385, 1136683, 1288688, 1122535]
    disliked_ids = [68495, 1685356, 111513, 25872, 221809, 1385063, 1079533, 129050, 209038, 38333, 1465469, 1244469, 55776, 238523, 342100, 1390968]
    tsne_embed_file = 'tsneperp=5_emb_matrix_female_5000_5-5_nonan.npy'
    data = {"bucket": bucket_name, "id_idx_map_file": id_idx_map_file, "liked_ids": liked_ids,
            "disliked_ids": disliked_ids, "tsne_embed_file": tsne_embed_file} 

    req = Mock(get_json=Mock(return_value=data), args=data) 
    response = main.send_image_viz(req) 
    json_obj = json.loads(response[0])          

    image_bytes = json_obj['image']
    decoded_image = base64.b64decode(image_bytes)
    image_io = BytesIO(decoded_image)
    pil_image = Image.open(image_io)


    assert pil_image.size == "True"
    print(response)
