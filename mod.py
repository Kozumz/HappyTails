import tensorflow as tf
try:
    model = tf.keras.models.load_model("pet_emotion.h5", compile=False)
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
