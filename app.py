from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
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

@app.route("/pet-emotion-predict", methods=['POST', 'GET'])
def pet_emotion_predictPage():
    pred = None
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((224,224))
                x = np.asarray(img)
                x = np.expand_dims(x, axis=0)
                x = x / 255.0

                # Usar el modelo cargado globalmente
                if model is None:
                    raise Exception("El modelo no se ha cargado correctamente.")

                pred = np.argmax(model.predict(x))
                print(pred)
        except Exception as e:
            print(f"Error during Recognition: {e}")
            message = "Error during recognition. Please try again."
            return render_template('index.html', message=message)
    return render_template('predict.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
