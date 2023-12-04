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

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


@csrf_exempt
def export_interruptibility_raw(request):
    app_data = fsm.db_get_documents_from_collection("Test_AppData")
    web_data = fsm.db_get_documents_from_collection("Test_WebApp")
    users = fsm.db_get_documents_from_collection("TestUsers")


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

    result["message"] = generate_message(persuasion_level)
    print("SERVER: " + persuasion_level)

    # Return the collection data as a JSON response
    return JsonResponse(result, safe=False)
