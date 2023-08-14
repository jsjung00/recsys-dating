from firebase_functions import firestore_fn, https_fn, options 

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore

app = initialize_app()

@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["get", "post"])
)
def getdocID(req):
    """Take the id parameter passed to this HTTP endpoint, get rating info from document, upload the recommendedIDs as a field"""
    # Grab the text parameter.
    docID = req.args.get("id")
    if docID is None:
        return https_fn.Response("No id parameter provided", status=400)

    firestore_client = firestore.client()

    doc_ref = firestore_client.collection("users-ratings").document(docID)
    doc_snapshot = doc_ref.get()

    if doc_snapshot.exists:
        rating_object = doc_snapshot.to_dict()
        likedIDs = rating_object['likedIDs']
        dislikedIDs = rating_object['dislikedIDs']
        


        return https_fn.Response(f"\n\n Liked IDs: {likedIDs}, Disliked IDs: {dislikedIDs}")
    else:
        return https_fn.Response("Document not found", status=404)
