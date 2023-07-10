import datetime
from firebase_admin import firestore
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import dirname.managers.firestoreManager as fsm
import dirname.managers.DataManager as dm
from dirname.inference_engine.FuzzySystem import FuzzySystem


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
def start_message_generation(request):
    WIN_SIZE_MINS = 2 * 60 * 24
    user_id = request.POST.get('userId')
    current_time = request.POST.get('currentTime')
    end_timestamp = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    start_timestamp = end_timestamp - datetime.timedelta(minutes=WIN_SIZE_MINS)

    print("StartTime", start_timestamp)
    print("EndTime", end_timestamp)

    data = fsm.db_get_emotions(user_id, start_timestamp, end_timestamp)
    data = dm.preprocess_data(data)

    fis = FuzzySystem()
    persuasion_level = fis.process_input(data)
    print("SERVER: " + persuasion_level)

    # Return the collection data as a JSON response
    return JsonResponse({"emotion_data": data, "persuasion_level": persuasion_level}, safe=False)
