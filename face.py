from keras.models import load_model
import cv2
import numpy as np

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Try different backends (CAP_MSMF or CAP_DSHOW) if the default fails
camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Grab the webcam's image
    ret, image = camera.read()

    if not ret:
        print("Error: Failed to capture image from the camera.")
        break

    # Resize the raw image into (224-height,224-width) pixels
    try:
        image_resized = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
    except cv2. error as e:
        print(f"OpenCV Error: {e}")
        break

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_array = (image_array / 127.5) - 1

    # Predict using the model
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print(f"Class: {class_name}, Confidence Score: {np.round(confidence_score * 100, 2)}%")

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the ESC key on your keyboard
    if keyboard_input == 27:
        print("Exiting program.")
        break

camera.release()
cv2.destroyAllWindows()