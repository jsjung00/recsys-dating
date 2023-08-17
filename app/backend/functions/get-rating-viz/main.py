import functions_framework
from google.cloud import storage 
import numpy as np 
import io 
import base64
from io import BytesIO
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import json 

def return_idxs(ids_arr, idxs_map_arr):
    #given an array of ids, returns list of corresponding row # in embedding matrix
    return [np.where(idxs_map_arr == id)[0][0] for id in ids_arr]

def create_plot(tsne_embeddings,idxs_map_arr, liked_ids, disliked_ids):
    liked_idxs = return_idxs(liked_ids,idxs_map_arr)
    disliked_idxs = return_idxs(disliked_ids,idxs_map_arr)
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

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return encoded_image


@functions_framework.http
def send_image_viz(request):
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
    if 'bucket' not in request_json or 'id_idx_map_file' not in request_json or 'liked_ids' not in request_json or \
    'disliked_ids' not in request_json or 'tsne_embed_file' not in request_json:
        return ("Missing parameters", 400, headers)
    
    bucket_name = request_json['bucket']
    id_idx_map_file = request_json['id_idx_map_file']
    liked_ids = request_json['liked_ids']
    disliked_ids = request_json['disliked_ids']
    tsne_embed_file = request_json['tsne_embed_file']

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    id_idx_map_blob, tsne_embed_blob = bucket.blob(id_idx_map_file), bucket.blob(tsne_embed_file)
    
    id_idx_map_bytes, tsne_embed_bytes = id_idx_map_blob.download_as_bytes(), tsne_embed_blob.download_as_bytes()
    id_idx_map = np.load(io.BytesIO(id_idx_map_bytes))
    tsne_embed = np.load(io.BytesIO(tsne_embed_bytes))

    encoded_image = create_plot(tsne_embed, id_idx_map, liked_ids, disliked_ids)
    response_data = {
        "image": encoded_image
    }
    return (json.dumps(response_data), 200, headers)





