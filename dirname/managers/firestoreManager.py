from firebase_admin import firestore
import pandas as pd


def db_get_documents_from_collection(collection_name):
    db = firestore.client()
    docs = db.collection(collection_name).get()
    # Create a list to store the document data
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    return pd.DataFrame.from_dict(data)


def db_get_emotions(user_id, start_timestamp, end_timestamp):
    db = firestore.client()

    # Fetch the collection data from Firebase
    docs = db.collection('Emotions').where('id_user', '==', user_id).where('timestamp', '>=', start_timestamp).where('timestamp', '<=', end_timestamp).get()

    # Create a list to store the document data
    data = []
    for doc in docs:
        # data.append([doc.get('emotion'), doc.get("source"), doc.get("value"), doc.get('timestamp')])
        data.append(doc.to_dict())

    return data


def db_get_training_data():
    db = firestore.client()
    docs = db.collection('TrainingData').get()
    # Create a list to store the document data
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    return pd.DataFrame.from_dict(data)
