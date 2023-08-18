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
from dirname.inference_engine.models import *


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
def start_message_generation(request, model_id):
    WIN_SIZE_MINS = 2 * 60 * 24
    user_id = request.POST.get('userId')
    current_time = request.POST.get('currentTime')
    end_timestamp = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    start_timestamp = end_timestamp - datetime.timedelta(minutes=WIN_SIZE_MINS)

    print("StartTime", start_timestamp)
    print("EndTime", end_timestamp)

    data = fsm.db_get_emotions(user_id, start_timestamp, end_timestamp)
    data = dm.preprocess_data(data)
    result = {"emotion_data": data}

    if model_id == FUZZY_MODEL:
        model = FuzzySystem()
    elif model_id == ENFS_MODEL:
        model = ENFS()
        model.load_model(ENFS_PATH)
        tmp = pd.DataFrame([data])
        data = np.array(tmp[FEATURES_COLS])
        data = np.array([data])
    elif model_id == SVM_MODEL:
        model = SVMmodel()
        model.load_model(SVM_PATH)
        tmp = pd.DataFrame([data])
        data = np.array(tmp[FEATURES_COLS])
    persuasion_level = model.process_input(data)
    result[LABEL_COL] = persuasion_level
    print("SERVER: " + persuasion_level)

    # Return the collection data as a JSON response
    return JsonResponse(result, safe=False)


@csrf_exempt
def train_model(request, model_id):
    data = fsm.db_get_training_data()
    #print(data)
    data = pd.read_csv(TRAINING_CSV_PATH)
    #print(data)
    X = np.array(data[FEATURES_COLS])
    y = np.array(data[LABEL_COL])
    if model_id == ENFS_MODEL:
        model = ENFS()
        model.train_model(X, y, 8, 15, ENFS_PATH)
        # model.test_model(X, y)
    elif model_id == SVM_MODEL:
        model = SVMmodel()
        model.train_model(X, y, SVM_PATH)
        # model.test_model(X, y)
    return JsonResponse({"message": "Model trained"})
