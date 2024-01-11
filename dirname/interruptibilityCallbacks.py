import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import dirname.managers.firestoreManager as fsm
import dirname.settings as settings
from dirname.inference_engine.SVMModel import SVMmodel
from dirname.inference_engine.models import *


@csrf_exempt
def export_interruptibility_raw(request):
    df_app_data = fsm.db_get_documents_from_collection("Test_AppData").sort_values(by=['id_user', 'delivery_time'], ascending=[True, False], na_position='first')
    df_web_data = fsm.db_get_documents_from_collection("Test_WebApp").sort_values(by=['id_user', 'timestamp'], ascending=[True, False], na_position='first')
    df_users = fsm.db_get_documents_from_collection("TestUsers").sort_values(by=['id'], ascending=[True])

    df_app_data = df_app_data.reset_index()
    next_idx = len(df_app_data)
    print(df_app_data, next_idx)
    print(df_web_data)
    new_indices = []
    curr_idx = 0
    MAX_DELAY_SECS = 60

    for index, row in df_app_data.iterrows():
        print("Evaluating " + str(index))
        while True:
            if row["id_user"] == df_web_data.iloc[curr_idx]["id_user"]:
                start_ts_ = row["delivery_time"]
                end_ts = df_web_data.iloc[curr_idx]["timestamp"]
                # Difference between two timestamps in seconds
                delta = end_ts - start_ts_
                if abs(delta.total_seconds()) <= MAX_DELAY_SECS:
                    print("ADD current index")
                    new_indices.append(index)
                    curr_idx += 1
                    break
                else:
                    if delta.total_seconds() > 0:   # It means that the current web register is less than the current row
                        print("ADD NEW index")
                        print(next_idx)
                        new_indices.append(next_idx)
                        next_idx += 1
                        curr_idx += 1
                    else:
                        break
            else:
                if row["id_user"] < df_web_data.iloc[curr_idx]["id_user"]:
                    break
                else:
                    new_indices.append(next_idx)
                    next_idx += 1
                    curr_idx += 1

    while curr_idx < len(df_web_data):
        new_indices.append(next_idx)
        next_idx += 1
        curr_idx += 1

    df_web_data.index = np.array(new_indices)
    print(df_web_data)

    df_all = pd.concat([df_app_data, df_web_data], axis=1)
    #df_all = df_all.drop(columns=['id_user_1', 'index'])

    df_all.to_csv(settings.STATIC_URL + "raw_interruptibility_data.csv")
    df_users.to_csv(settings.STATIC_URL + "raw_users_data.csv")

    # Return the collection data as a JSON response
    return JsonResponse({"message": "Data exported"}, safe=False)


def get_training_data():
    # After manual cleaning of data: removing null rows and remove when the E4 was not connected.
    df_data = pd.read_csv(settings.STATIC_URL + "raw_interruptibility_data_processed.csv", delimiter=";")
    df_data = df_data[df_data["id_user"].notnull()]
    # Eliminamos esto mas para entrenar con todos los campos
    df_data = df_data[df_data["id_user_1"].notnull()]

    # interruptibility: 0 = NO, 1 = YES
    df_data["label"] = 0
    df_data.loc[df_data["interruptibility_level"] > 2, "label"] = 1

    # {'fearful': 0.00828965167356597, 'disgusted': 0.0010895020867715867, 'angry': 0.03830508180803599, 'sad': 0.3768521692077576, 'surprised': 0.002607696335718302, 'neutral': 0.5548311083724625, 'happy': 0.018024786340686806}
    df_data[['tt', 'fearful', 'disgusted', 'angry', 'sad', 'surprised', 'neutral', 'happy']] = df_data['emotions'].str.split(r"\'\w+\':\s", expand=True)
    df_data['fearful'] = df_data['fearful'].str.replace(', ', '')
    df_data['disgusted'] = df_data['disgusted'].str.replace(', ', '')
    df_data['angry'] = df_data['angry'].str.replace(', ', '')
    df_data['sad'] = df_data['sad'].str.replace(', ', '')
    df_data['surprised'] = df_data['surprised'].str.replace(', ', '')
    df_data['neutral'] = df_data['neutral'].str.replace(', ', '')
    df_data['happy'] = df_data['happy'].str.replace('}', '')

    print(df_data[['tt', 'fearful', 'disgusted', 'angry', 'sad', 'surprised', 'neutral', 'happy', "interruptibility_level", "label"]])
    return df_data


@csrf_exempt
def train_for_interruptibility(request, model_id):
    data = get_training_data()
    #COLS_TRAIN = [stress;reaction_time;id_user;physical_activity;airplane_mode;surrounding_sound;screen;battery_level;completion_time;day_of_week;mobile_data;delivery_time;wifi;interruptibility_level;stress_level;charge_status;notification_ringtone;current_activity;priority_current_activity;id_user_1;attention_level;emotions;timestamp;interaction_others]
    COLS_TRAIN = ["stress", "physical_activity", "airplane_mode", "surrounding_sound", "screen", "battery_level", "mobile_data", "wifi", "stress_level", "charge_status", "notification_ringtone", "current_activity", "priority_current_activity", "attention_level"]
    COL_LABEL = "label"
    X = np.array(data[COLS_TRAIN])
    y = np.array(data[COL_LABEL])
    if model_id == SVM_MODEL:
        model = SVMmodel()
        model.train_model(X, y, MODELS_PATH + "svm_interruptibility_config")
        model.test_model(X, y)
    return JsonResponse({"message": "Model trained"})
