import datetime
import numpy as np
import pandas as pd
from firebase_admin import firestore
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import dirname.managers.firestoreManager as fsm
import dirname.managers.DataManager as dm
from dirname.inference_engine.FuzzySystem import FuzzySystem
from dirname.inference_engine.SVMModel import SVMmodel
from dirname.inference_engine.enfs.enfs import ENFS
from dirname.config_vars import *

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


def firebase_collection(request):
    # Access the Firestore database
    db = firestore.client()

    # Fetch the collection data from Firebase
    collection_ref = db.collection('TestCollection')
    docs = collection_ref.get()

    # Create a list to store the document data
    data = []
    for doc in docs:
        data.append(doc.to_dict())

    # Return the collection data as a JSON response
    return JsonResponse(data, safe=False)


@csrf_exempt
def get_emotions(request):
    user_id = request.POST.get('userId')
    start_date = request.POST.get('startDate')
    end_date = request.POST.get('endDate')

    # convert start_date and end_date to timestamp
    start_timestamp = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_timestamp = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    # Create a list to store the document data
    data = fsm.db_get_emotions(user_id, start_timestamp, end_timestamp)

    # Return the collection data as a JSON response
    return JsonResponse(data, safe=False)


@csrf_exempt
def upload_training_data(request):
    data_df = pd.read_csv(TRAINING_CSV_PATH, index_col=0)
    data = data_df.to_dict("records")
    db = firestore.client()
    for val in data:
        db.collection("TrainingData").add(val)
    return JsonResponse({"message": "Data was inserted", "data": data})


def generate_message(persuasion_level):
    # Infer the modality
    # Generate message object
    import random
    num = random.random()
    if num > 0.65:
        return {"type": "visual", "value": "red", "time": 3}
    elif num > 0.3:
        return {"type": "audio",
                "value": "https://drive.google.com/uc?export=download&id=13lrUpMAiqiznF9kk3LULLWtH3nf3ioAz", "time": 2}
    else:
        return {"type": "speech",
                "value": "Por favor toma un descanso", "time": 1}


@csrf_exempt
def start_message_generation(request):
    user_id = request.POST.get('userId')
    current_time = request.POST.get('currentTime')
    end_timestamp = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    start_timestamp = end_timestamp - datetime.timedelta(minutes=int(os.environ.get('MINS_FOR_PERSUASION_INFERENCE')))

    print("StartTime", start_timestamp)
    print("EndTime", end_timestamp)

    data = fsm.db_get_emotions(user_id, start_timestamp, end_timestamp)
    data = dm.preprocess_data(data)
    result = {"emotion_data": data}

    model_id = os.environ.get('MODEL_PERSUASION_INFERENCE')

    if model_id == ENFS_MODEL:
        model = ENFS()
        model.load_model(MODELS_PATH + "enfs_config")
        tmp = pd.DataFrame([data])
        data = np.array(tmp[FEATURES_COLS])
        data = np.array([data])
    elif model_id == SVM_MODEL:
        model = SVMmodel()
        model.load_model(MODELS_PATH + "svm_config")
        tmp = pd.DataFrame([data])
        data = np.array(tmp[FEATURES_COLS])
    else:
        model = FuzzySystem()

    # inferring the persuasion level
    persuasion_level = model.process_input(data)
    result[LABEL_COL] = persuasion_level

    # generating the corresponding message
    result["message"] = generate_message(persuasion_level)
    print("SERVER: " + persuasion_level)

    # Return the collection data as a JSON response
    return JsonResponse(result, safe=False)


@csrf_exempt
def train_model(request, model_id):
    data = fsm.db_get_training_data()
    X = np.array(data[FEATURES_COLS])
    y = np.array(data[LABEL_COL])
    if model_id == ENFS_MODEL:
        model = ENFS()
        model.train_model(X, y, 8, 15, MODELS_PATH + "enfs_config")
        # model.test_model(X, y)
    elif model_id == SVM_MODEL:
        model = SVMmodel()
        model.train_model(X, y, MODELS_PATH + "svm_config")
        # model.test_model(X, y)
    return JsonResponse({"message": "Model trained"})
