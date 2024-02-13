import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import dirname.managers.firestoreManager as fsm
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


def