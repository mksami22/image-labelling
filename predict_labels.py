from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np



model = tf.saved_model.load('./model/')
jewel_model = tf.keras.models.load_model('./model/', custom_objects={'KerasLayer':hub.KerasLayer})

img = []

img.append([[['ex.png']]])

print(img)

predictions = jewel_model.predict(img)

print(predictions)



