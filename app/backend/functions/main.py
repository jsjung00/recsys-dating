# Deploy with `firebase deploy`
import pandas as pd 
from pathlib import Path 
from firebase_functions import https_fn, firestore_fn
from firebase_admin import initialize_app, firestore, credentials


def upload_clusters_file(clusters_file):
    clusters_df = pd.read_csv(clusters_file, index_col=0)
    doc_name = Path(clusters_file).stem 
    cluster_labels = clusters_df.iloc[:,0]
    tmdb_IDs = clusters_df.iloc[:,1]
    cluster_index_groups = {} #key is cluster number, values is TMDB_ID
    for i in range(0, len(cluster_labels)):
        if str(cluster_labels[i]) in cluster_index_groups:
            cluster_index_groups[str(cluster_labels[i])].append(int(tmdb_IDs[i]))
        else:
            cluster_index_groups[str(cluster_labels[i])] = [int(tmdb_IDs[i])]

    db.collection("clusters").document(doc_name).set(cluster_index_groups)
    return 

if __name__ == "__main__":

    cred = credentials.Certificate("../../NO-UPLOAD/account_file.json")
    app = initialize_app(cred)
    
    db = firestore.client()
    upload_clusters_file("../../../files/TMDBID_KMEANS_k=10_female_5000_5-5_nonan.csv")
    upload_clusters_file("../../../files/TMDBID_KMEANS_k=10_male_5000_5-5_nonan.csv")