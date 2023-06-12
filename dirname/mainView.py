import firebase_admin
import datetime
from firebase_admin import credentials, firestore
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json

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
    start_date = request.POST.get('startDate')
    end_date = request.POST.get('endDate')
    # convert start_date and end_date to timestamp
    start_timestamp = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_timestamp = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    db = firestore.client()

    # Fetch the collection data from Firebase
    collection_ref = db.collection('TestCollection')
    # docs = collection_ref.get()
    docs = db.collection('Emotions').where('timestamp', '>=', start_timestamp).where('timestamp', '<=', end_timestamp).get()

    # Create a list to store the document data
    data = []
    for doc in docs:
        data.append([doc.get('emotion'), doc.get('timestamp')])

    # Return the collection data as a JSON response
    return JsonResponse(data, safe=False)
