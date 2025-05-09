from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

# Cargar el modelo FUERA de las funciones de solicitud
try:
    model = load_model("pet_emotion.h5", compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None  # Asegurar que el modelo esté definido incluso si hay un error

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/pet-emotion", methods=['GET', 'POST'])
def PetEmotionPage():
    return render_template('index.html')

def save_prediction(emotion, animal_type):
    conn = sqlite3.connect('predicciones.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO predicciones (emocion, fecha, tipo_animal) VALUES (?, ?, ?)', 
                   (emotion, datetime.now(), animal_type))
    conn.commit()
    conn.close()

@app.route("/pet-emotion-predict", methods=['POST', 'GET'])
def pet_emotion_predictPage():
    pred = None
    emotion = None
    animal_type = None
    if request.method == 'POST':
        try:
            if 'image' in request.files and 'animalType' in request.form:
                animal_type = request.form['animalType']
                img = Image.open(request.files['image'])
                img = img.resize((224,224))
                x = np.asarray(img)
                x = np.expand_dims(x, axis=0)
                x = x / 255.0

                if model is None:
                    raise Exception("El modelo no se ha cargado correctamente.")

                pred = np.argmax(model.predict(x))
                emotions = ["ANGRY", "HAPPY", "RELAXED", "SAD"]
                emotion = emotions[pred]
                save_prediction(emotion, animal_type)  # Guardar en la base de datos
        except Exception as e:
            print(f"Error during Recognition: {e}")
            message = "Error during recognition. Please try again."
            return render_template('index.html', message=message)
    return render_template('predict.html', pred=pred)

@app.route("/predictions", methods=['GET'])
def show_predictions():
    conn = sqlite3.connect('predicciones.db')
    cursor = conn.cursor()
    cursor.execute('SELECT tipo_animal, emocion, fecha FROM predicciones ORDER BY fecha DESC')
    data = cursor.fetchall()
    conn.close()
    return render_template('predictions.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
