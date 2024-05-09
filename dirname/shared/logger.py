from firebase_admin import firestore
from firebase_admin.firestore import SERVER_TIMESTAMP

# Global variable to hold the collection name
LOG_EVENTS_COLLECTION = "log_events"


def initialize_firestore():
    """Initializes the Firestore client."""
    return firestore.client()


def send_log_to_firestore(event_name, input_params, data_processed, response, db=None):
    """
    Sends a log event to a Firestore collection.

    Args:
        event_name (str): The name of the event to log.
        input_params (dict): Parameters related to the input data.
        data_processed (str or dict): The processed data details.
        response (str or dict): The response or output data.
        db (firestore.Client, optional): Firestore client instance. If not provided, a new one will be created.

    Returns:
        str: ID of the newly added log event.
    """
    try:
        # Initialize Firestore client if not provided
        if db is None:
            db = initialize_firestore()

        # Build the log event dictionary
        log_event = {
            "event": event_name,
            "input_params": input_params,
            "data_processed": data_processed,
            "output": response,
            "created_at": SERVER_TIMESTAMP
        }

        # Add log to Firestore and obtain reference
        log_ref = db.collection(LOG_EVENTS_COLLECTION).add(log_event)[1]

        print(f"Successfully added log event with ID {log_ref.id}")
        return log_ref.id
    except Exception as e:
        print(f"Error while logging event: {e}")
        return None
