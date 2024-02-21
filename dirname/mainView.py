import datetime
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
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


def generate_message(user_id, current_time, persuasion_level):
    # Infer the modality
    user_state = fsm.db_get_interruptibility_data(user_id, current_time)
    message_obj = fsm.db_get_persuasive_messages(persuasion_level)

    modality = "audio"
    print(user_state)
    if user_state["attention_level"][0] == 3 or user_state["stress_level"][0] == 5:
        modality = "color"

    # Generate message object
    if modality == "color":
        return {"type": "visual", "value": message_obj[modality], "time": 3}
    elif modality == "audio":
        idx = random.randint(0, NUM_AVAILABLE_MESSAGES - 1)
        return {"type": "audio",
                "value": message_obj["messages"][idx], "time": 1}


@csrf_exempt
def generate_persuasive_message(request):
    user_id = request.POST.get('userId')
    current_time = request.POST.get('currentTime') if request.POST.get('currentTime') else datetime.now()
    end_timestamp = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S %Z%z')
    start_timestamp = end_timestamp - timedelta(minutes=int(os.environ.get('MINS_FOR_PERSUASION_INFERENCE')))

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
    result["message"] = generate_message(user_id, end_timestamp, persuasion_level)
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
