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
















# TASK=2                                                                                                                                                           # Detailed Report on Python-based Face Recognition Attendance System
![WhatsApp Image 2025-01-10 at 12 03 56 PM](https://github.com/user-attachments/assets/897d3738-6ead-47cb-b6ac-015a75ad8d63)


## 1. Importing Required Libraries
```python
import cv2
import numpy as np
import mysql.connector
from tensorflow.keras.models import load_model
import time
```
**Explanation:**
- `cv2`: Used for image capture and processing via the webcam.
- `numpy`: Handles numerical operations, especially for image array manipulation.
- `mysql.connector`: Connects Python to the MySQL database for attendance records.
- `load_model`: Loads the pre-trained Keras model for face recognition.
- `time`: Manages time-related functions, especially for cooldown periods.![Screenshot 2025-01-14 111956](https://github.com/user-attachments/assets/bce5d518-ceab-4e39-b329-eb434b63fce3)


---

## 2. Loading the Trained Model
```python
try:
    model = load_model("keras_model.h5", compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
```
**Explanation:**
- Loads a pre-trained model (`keras_model.h5`).
- If successful, prints confirmation. If not, it prints the error and exits.

---

## 3. Loading Labels
```python
try:
    class_names = open("labels.txt", "r").readlines()
    print("Labels loaded successfully.")
except Exception as e:
    print(f"Error loading labels: {e}")
    exit()
```
**Explanation:**
- Reads the `labels.txt` file, which contains class names (e.g., student names).
- If loading fails, prints the error and stops execution.

---

## 4. Connecting to MySQL Database
```python
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="AttendanceSystem"
    )
    c = conn.cursor()
    print("Connected to MySQL database.")
except mysql.connector.Error as err:
    print(f"Database connection failed: {err}")
    exit()
```
**Explanation:**
- Establishes a connection to the `AttendanceSystem` database.
- Uses `root` credentials for local connection.
- Initializes a cursor (`c`) for executing SQL commands.

---

## 5. Defining Attendance Recording Function
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
**Explanation:**
- Checks if the person (`name`) already has attendance marked.
- If not, inserts their name with status 'present' into the `attendance` table.
- If already present, it notifies the user.![Screenshot 2025-01-14 115103](https://github.com/user-attachments/assets/11bb4eac-5c6f-4c63-b0f2-2298b4aa0869)


---

## 6. Initializing Webcam
```python
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```
**Explanation:**
- Activates the webcam using `cv2.VideoCapture`.
- `cv2.CAP_DSHOW` ensures compatibility with Windows.![Screenshot 2025-01-14 105317](https://github.com/user-attachments/assets/2271d240-a01e-435a-9b48-f035d2a64c22)


---

## 7. Checking Webcam Access
```python
if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()
```
**Explanation:**
- Verifies webcam availability. Exits if not accessible.![Screenshot 2025-01-10 113652](https://github.com/user-attachments/assets/dcaebb58-f892-4ad4-b4ed-62d31c24b7b8)


---

## 8. Defining Control Variables
```python
last_recognized_name = None
confidence_threshold = 0.95
cooldown_time = 3
last_recognition_time = time.time()
```
**Explanation:**
- `last_recognized_name`: Stores the last recognized person.
- `confidence_threshold`: Sets a 95% confidence limit for accurate recognition.
- `cooldown_time`: Defines a 3-second gap between recognitions.
- `last_recognition_time`: Tracks the last recognition timestamp.

---

## 9. Main Loop for Face Recognition
```python
while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to capture image. Check your webcam.")
        break
```
**Explanation:**
- Captures continuous frames from the webcam.
- If frame capture fails, the loop breaks.

---

## 10. Preprocessing Captured Image
```python
image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
image_array = (image_array / 127.5) - 1
```
**Explanation:**
- Resizes the image to `224x224` (model input size).
- Normalizes pixel values between -1 and 1 for model prediction.

---

## 11. Making Predictions
```python
prediction = model.predict(image_array, verbose=0)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]
```
**Explanation:**
- Predicts the face class.
- Identifies the label with the highest probability.
- Extracts the class name and confidence score.

---

## 12. Attendance Logic
```python
if confidence_score >= confidence_threshold and class_name != last_recognized_name and current_time - last_recognition_time >= cooldown_time:
    record_attendance(class_name)
    last_recognized_name = class_name
    last_recognition_time = current_time
```
**Explanation:**
- Marks attendance if confidence is high, the person isn't recently recognized, and the cooldown has passed.![Screenshot 2025-01-14 115704](https://github.com/user-attachments/assets/3eacee31-64c7-41a6-bf7c-9fa5ed3a84f6)


---

## 13. Displaying Results
```python
cv2.putText(image, f"Detected: {class_name} ({confidence_score * 100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imshow("Attendance System", image)
```
**Explanation:**
- Overlays detected name and confidence score on the webcam feed.![Screenshot 2025-01-14 115704](https://github.com/user-attachments/assets/afb828d9-5cba-4b19-a867-f574dbc5f130)


---

## 14. Exiting the Program
```python
if cv2.waitKey(1) & 0xFF == 27:
    break
```
**Explanation:**
- Breaks the loop when the ESC key is pressed.

---

## 15. Cleanup
```python
camera.release()
cv2.destroyAllWindows()
conn.close()
```
**Explanation:**
- Releases the webcam.
- Closes all OpenCV windows.
- Closes the MySQL connection.

---

## Conclusion
This program effectively integrates face recognition with a MySQL database to automate attendance tracking. It utilizes machine learning for real-time recognition, ensuring efficient and accurate attendance management.

