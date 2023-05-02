from flask import Flask, jsonify, request
from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://franslom:979446642@cluster0.4xla9g9.mongodb.net/?retryWrites=true&w=majority")
db = cluster["thesisDB"]
collection = db["Emotion"]

app = Flask(__name__)


@app.route("/")
def index():
    return "<p>Holaaaaa</p>"


@app.route('/insert/<emotion>/<value>')
def insert_emotion_value(emotion, value):
    print("insert:", emotion, value)
    collection.insert_one({"stress": float(value),
                           "facial_emotion": {"angry": 0, "disgust": 0, "fear": 0, "happy": 0,
                                              "sad": 0, "surprise": 0, "neutral": 0}, "date": "hoy",
                           "id_user": "63e51674"})
    return jsonify({"result": "success"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)