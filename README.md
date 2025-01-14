#  TASK=1    Face Recognition Python 


![Screenshot 2025-01-14 104543](https://github.com/user-attachments/assets/95eb663c-7164-4cdb-bb3e-daa6bfa3e73c)
![Screenshot 2025-01-14 104605](https://github.com/user-attachments/assets/39bd5eb7-ea1d-4c48-9cef-b6392cf36851)
![Screenshot 2025-01-14 104634](https://github.com/user-attachments/assets/8f48e840-c91f-4182-b912-abf2224ef742)
![Screenshot 2025-01-14 104824](https://github.com/user-attachments/assets/edee524e-3d09-4817-a8fa-a1f5c261c81a)
![Screenshot 2025-01-14 104932](https://github.com/user-attachments/assets/df29008e-9b4d-461f-a871-2662eea4160e)

## 1. Importing Required Libraries
```python
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
import os
```
- **load_model**: Used to load a pre-trained Keras model saved in the `keras_model.h5` file.
- **cv2**: OpenCV library for real-time computer vision tasks like capturing and processing images.
- **numpy**: For numerical operations, especially for handling image data in array format.
- **tensorflow**: Checks hardware resources (e.g., GPU) and supports deep learning computations.
- **os**: Interacts with the operating system to configure environment variables.

- ![Screenshot 2025-01-14 105226](https://github.com/user-attachments/assets/52ddd70c-b051-40c4-a098-7d83584f2c34)


## 2. Checking GPU Availability
```python
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
- Prints the number of GPUs available for TensorFlow processing.

## 3. Disabling GPU (Force CPU Usage)
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```
- Forces the program to use the CPU by disabling the GPU.

## 4. Configuring NumPy and Loading the Model
```python
np.set_printoptions(suppress=True)
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
```
- **np.set_printoptions(suppress=True)**: Prevents scientific notation in NumPy outputs.
- **model**: Loads the pre-trained model for prediction.
- **class_names**: Reads class labels from `labels.txt` for result interpretation.

## 5. Accessing the Webcam
```python
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```
- Initializes the webcam using the DirectShow backend (`CAP_DSHOW`).
- 

### Error Handling for Camera Access
```python
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()
```
- Checks if the webcam is accessible. If not, it prints an error and stops the program.

## 6. Continuous Frame Capture Loop
```python
while True:
    ret, image = camera.read()
```
- Continuously captures frames from the webcam.
- **ret**: Boolean indicating if the frame was captured successfully.
- **image**: Captured frame.

### Error Handling for Frame Capture
```python
    if not ret:
        print("Error: Failed to capture image from the camera.")
        break
```
- Stops the loop if frame capture fails.

### Resizing Captured Image
```python
    try:
        image_resized = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
        break
```
- Resizes the image to `224x224` pixels to fit the model's input size.
- Uses a `try-except` block to handle errors during resizing.
- 

### Display Webcam Feed
```python
    cv2.imshow("Webcam Image", image)
```
- Displays the live webcam feed in a window titled "Webcam Image".
- ![Screenshot 2025-01-14 105317](https://github.com/user-attachments/assets/513e0081-b56d-4385-b80d-2c82c08b2d5a)


### Preparing Image for Model Prediction
```python
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1
```
- Converts the resized image to a NumPy array with shape `(1, 224, 224, 3)`.
- Normalizes the pixel values to the range `[-1, 1]` for model compatibility.

### Making Predictions
```python
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
```
- **model.predict**: Makes a prediction on the processed image.
- **np.argmax**: Identifies the index of the highest prediction probability.
- **class_name**: Maps the predicted index to a label.
- **confidence_score**: Retrieves the prediction confidence.
- ![Screenshot 2025-01-14 105334](https://github.com/user-attachments/assets/88d59cd4-32b9-445c-82d0-51fd02fa34ab)


### Displaying Prediction Results
```python
    print(f"Class: {class_name}, Confidence Score: {np.round(confidence_score * 100, 2)}%")
```
- Prints the predicted class and its confidence percentage in the console.
- ![Screenshot 2025-01-14 105334](https://github.com/user-attachments/assets/ce165199-a8f6-480e-9008-058fcf23d06e)


### Exiting the Program with ESC Key
```python
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        print("Exiting program.")
        break
```
- **cv2.waitKey(1)**: Waits for keyboard input.
- **27**: ASCII code for the ESC key. Ends the loop and exits if pressed.

## 7. Releasing Resources
```python
camera.release()
cv2.destroyAllWindows()
```
- **camera.release()**: Frees the webcam.
- **cv2.destroyAllWindows()**: Closes all OpenCV windows.

# Conclusion
This Python script captures real-time webcam footage, processes it, and predicts the class of the object/person using a pre-trained model. Results are printed to the console with a confidence score, and the webcam feed is displayed live. Press ESC to stop the program.
















 # TASK=2     Face Recognition-Based Attendance System: Detailed Code Explanation

## Introduction
This project implements an automated attendance system using face recognition. It integrates a pre-trained deep learning model for face detection, OpenCV for real-time image capture, and MySQL for recording attendance.
![WhatsApp Image 2025-01-10 at 12 03 56 PM](https://github.com/user-attachments/assets/c633b523-e435-4670-bdc7-598c0fa086df)


## Detailed Line-by-Line Explanation

### 1. Importing Libraries
```python
import cv2
import numpy as np
import mysql.connector
from tensorflow.keras.models import load_model
import time
```
- **cv2**: OpenCV library for real-time computer vision tasks.
- **numpy**: Used for numerical operations, especially array manipulation.
- **mysql.connector**: Connects Python with the MySQL database.
- **load_model**: Loads the pre-trained Keras model for face recognition.
- **time**: Provides time-related functions (e.g., for cooldown logic).
![Screenshot 2025-01-14 111956](https://github.com/user-attachments/assets/3d322407-ab28-4c25-92f9-0bacc39d052d)


### 2. Loading the Trained Model
```python
model = load_model("keras_model.h5", compile=False)
```
- Loads a pre-trained Keras model (`keras_model.h5`) without compiling since it's used only for prediction.

### 3. Loading Class Labels
```python
class_names = open("labels.txt", "r").readlines()
```
- Reads the class labels from `labels.txt` to map predictions to user names.
![Screenshot 2025-01-14 112044](https://github.com/user-attachments/assets/7854d0bf-3516-4002-9817-00c57d336c9a)


### 4. Connecting to MySQL Database
```python
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="AttendanceSystem"
)
c = conn.cursor()
```
- Establishes a connection to the local MySQL database `AttendanceSystem` using the root credentials.
- Creates a cursor `c` to execute SQL queries.

### 5. Function to Record Attendance
```python
def record_attendance(name):
    try:
        c.execute("SELECT * FROM attendance WHERE name = %s", (name,))
        if not c.fetchone():
            c.execute("INSERT INTO attendance (name, status) VALUES (%s, %s)", (name, 'present'))
            conn.commit()
            print(f"Attendance marked for {name}")
        else:
            print(f"Attendance already marked for {name}")
    except Exception as e:
        print(f"Error inserting attendance: {e}")
```
- **Checks** if the user's attendance is already marked.
- **Inserts** a new record if not, marking the user as 'present'.
- **Commits** the change to the database.
![Screenshot 2025-01-14 113130](https://github.com/user-attachments/assets/d8590d8a-1ef4-4fdf-a901-3a33abf85067)


### 6. Accessing the Webcam
```python
camera = cv2.VideoCapture(0)
```
- Starts capturing video from the default webcam (`0`).
![Screenshot 2025-01-10 113532](https://github.com/user-attachments/assets/64006a6d-824b-411e-b77c-352f5e7ec719)


### 7. Control Variables for Detection
```python
last_recognized_name = None
confidence_threshold = 0.95
cooldown_time = 3
last_recognition_time = time.time()
```
- **last_recognized_name**: Stores the last recognized name to avoid repeated entries.
- **confidence_threshold**: Minimum model confidence (95%) required for recognition.
- **cooldown_time**: Wait time (3 seconds) between recognitions.
- **last_recognition_time**: Records the last recognition timestamp.
![Screenshot 2025-01-10 110324](https://github.com/user-attachments/assets/98de4687-a28a-4800-9bae-f71f28b09bb5)


### 8. Real-Time Detection Loop
```python
while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to capture image. Check your webcam.")
        break
```
- Captures frames from the webcam.
- Exits the loop if the camera fails.

### 9. Image Preprocessing
```python
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1
```
- **Resizes** the captured image to 224x224 pixels (model input size).
- **Converts** the image to a NumPy array and reshapes it.
- **Normalizes** pixel values between -1 and 1 for model compatibility.

### 10. Model Prediction
```python
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
```
- **Predicts** the class using the model.
- **Finds** the index of the highest probability class.
- **Maps** the index to a label (user name).
- **Extracts** the prediction confidence score.

### 11. Attendance Marking Logic
```python
    current_time = time.time()
    if confidence_score >= confidence_threshold and class_name != last_recognized_name and current_time - last_recognition_time >= cooldown_time:
        print(f"Recognized as: {class_name} with confidence {confidence_score*100:.2f}%")
        record_attendance(class_name)
        last_recognized_name = class_name
        last_recognition_time = current_time
```
- **Validates** recognition based on:
  - Confidence threshold.
  - Different from the last recognized name.
  - Cooldown time elapsed.
- **Marks attendance** and updates tracking variables.

### 12. Displaying Recognition on Webcam Feed
```python
    cv2.putText(image, f"Detected: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Attendance System", image)
```
- **Overlays** the recognized name on the webcam feed.
- **Displays** the live video feed.

### 13. Exit Condition
```python
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break
```
- **Listens** for the ESC key (`27`) to terminate the program.

### 14. Cleanup
```python
camera.release()
cv2.destroyAllWindows()
conn.close()
```
- **Releases** the webcam resource.
- **Closes** OpenCV windows.
- **Disconnects** from the MySQL database.

## Conclusion
This Python script integrates face recognition and database management for a real-time attendance system. It efficiently captures images, identifies users, and records attendance securely in a MySQL database.

