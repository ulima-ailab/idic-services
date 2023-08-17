import dirname.settings as settings

FUZZY_MODEL = "fuzzy"
KMEANS_MODEL = "kmeans"
ENFS_MODEL = "enfs"

MODELS_PATH = settings.STATIC_URL + "models/"
ENFS_PATH = MODELS_PATH + "enfs_config"
KMEANS_PATH = MODELS_PATH + "kmeans_config.txt"

TRAINING_CSV_PATH = settings.STATIC_URL + "training_data.csv"
