import dirname.settings as settings

FEATURES_COLS = ["stress", "angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
LABEL_COL = "persuasion_level"

FUZZY_MODEL = "fuzzy"
SVM_MODEL = "svm"
ENFS_MODEL = "enfs"

MODELS_PATH = settings.STATIC_URL + "models/"

TRAINING_CSV_PATH = settings.STATIC_URL + "training_data.csv"

NUM_AVAILABLE_MESSAGES = 6
