import requests
import pandas as pd


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
    # print(result)
    return result
