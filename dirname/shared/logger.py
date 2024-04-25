from firebase_admin import firestore

LOG_EVENTS_COLLECTION = "log_events"


def send_log_firestore(event_name, input_params, data_processed, response):
    db = firestore.client()
    reg_log = {
        "event": event_name, "input_params": input_params,
        "data_processed": data_processed, "output": response
    }
    update_time, log_ref = db.collection(LOG_EVENTS_COLLECTION).add(reg_log)
    log_ref = db.collection(LOG_EVENTS_COLLECTION).document(log_ref.id)
    log_ref.update({"created_at": firestore.SERVER_TIMESTAMP})
    print(f"Added log event with id {log_ref.id}")
