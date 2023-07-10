import random

import requests
import pandas as pd
import numpy as np


def recover_data_by_user(user_id, start_date, end_date):
    # Recover data from DB
    url = 'http://127.0.0.1:8000/get-emotions/'
    #req_data = {"userId": user_id, "startDate": "2023-01-01 00:00:00", "endDate": "2024-07-31 00:00:00"}
    req_data = {"userId": user_id, "startDate": start_date, "endDate": end_date}
    response = requests.post(url, data=req_data)
    data = response.json()
    print(data)
    # Process data to generate expected structure (pandas)


def preprocess_data(input_data):
    df = pd.DataFrame.from_records(input_data, columns=["emotion", "source", "value"])
    grouped_df = df.groupby(["emotion", "source"])
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()
    print(mean_df)
    result = {}
    for ind in mean_df.index:
        result[mean_df["emotion"][ind]] = mean_df["value"][ind]
    print(result)
    return result


class DataManager:
    def __init__(self):
        print("Se crea el modelo")

    def get_temp_data(self):
        import random
        cols = ["p_stress", "f_angry", "f_disgusted", "f_fearful", "f_happy", "f_sad", "f_surprised", "f_neutral"]
        size = 15
        p_stress = [random.randint(1, 5) for i in range(0, size)]
        f_angry = [random.random() for i in range(0, size)]
        f_disgusted = [random.random() for i in range(0, size)]
        f_fearful = [random.random() for i in range(0, size)]
        f_happy = [random.random() for i in range(0, size)]
        f_sad = [random.random() for i in range(0, size)]
        f_surprised = [random.random() for i in range(0, size)]
        f_neutral = [random.random() for i in range(0, size)]
        df1 = pd.DataFrame(np.array([p_stress, f_angry, f_disgusted, f_fearful, f_happy, f_sad, f_surprised, f_neutral]), columns=cols)
        return df1

    def train_model(self):
        data = self.get_temp_data()
        print(data)
        return {"result": True}


if __name__ == '__main__':
    model = DataManager()
    out = model.train_model()
    print(out)
