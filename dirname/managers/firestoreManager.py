from firebase_admin import firestore
import pandas as pd
from datetime import datetime, timedelta
import pytz


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


def db_get_documents_by_range_time(collection_name, user_id, start_timestamp, end_timestamp, ascending=True):
    db = firestore.client()
    docs = (db.collection(collection_name).where('id_user', '==', user_id).where('timestamp', '>=', start_timestamp)
            .where('timestamp', '<=', end_timestamp).get())
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp", ascending=ascending)
    return df


def db_get_interruptibility_data(user_id, current_timestamp):
    start_timestamp = current_timestamp - timedelta(minutes=int(5))
    print("STARTTT", start_timestamp)

    app_cols = ["surrounding_sound", "stress_level", "physical_activity"]
    web_cols = ["attention_level"]
    emo_cols = ["neutral", "sad", "disgusted", "stress", "happy"]

    final_data = {}
    for key in (app_cols + web_cols + emo_cols):
        final_data[key] = -1

    # Recovering data from the context_app (android)
    app_cols = app_cols + ["stress"]
    df = db_get_documents_by_range_time("Context_app", user_id, start_timestamp, current_timestamp, False)
    if len(df.columns) > 0:
        df = df[app_cols]
        final_data.update(df.to_dict('records')[0])

    # Recovering data from context_web
    df = db_get_documents_by_range_time("Context_web", user_id, start_timestamp, current_timestamp, False)
    print(df)
    if len(df.columns) > 0:
        df = df[web_cols]
        final_data.update(df.to_dict('records')[0])

    # Recovering data from emotions
    df = db_get_documents_by_range_time("Emotions", user_id, start_timestamp, current_timestamp, False)
    print(df)
    if len(df.columns) > 0:
        df = df.iloc[0:7]
        print(df)
        for idx in df.index:
            if df['emotion'][idx] in emo_cols:
                final_data[df['emotion'][idx]] = df['value'][idx]

    return pd.DataFrame([final_data])
