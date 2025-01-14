import cv2
import numpy as np
import mysql.connector
from tensorflow.keras.models import load_model
import time

# Load the model
try:
    model = load_model("keras_model.h5", compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the labels
try:
    class_names = open("labels.txt", "r").readlines()
    print("Labels loaded successfully.")
except Exception as e:
    print(f"Error loading labels: {e}")
    exit()

# Connect to MySQL database
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",  # Replace with your MySQL password
        database="AttendanceSystem"
    )
    c = conn.cursor()
    print("Connected to MySQL database.")
except mysql.connector.Error as err:
    print(f"Database connection failed: {err}")
    exit()

# Function to record attendance
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

# Initialize webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Variables for control
last_recognized_name = None
confidence_threshold = 0.95
cooldown_time = 3  # Cooldown in seconds
last_recognition_time = time.time()

while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to capture image. Check your webcam.")
        break

    # Resize and normalize image
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    # Predict
    try:
        prediction = model.predict(image_array, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
    except Exception as e:
        print(f"Prediction error: {e}")
        continue

    current_time = time.time()

    # Record attendance if conditions are met
    if confidence_score >= confidence_threshold and class_name != last_recognized_name and current_time - last_recognition_time >= cooldown_time:
        print(f"Recognized as: {class_name} with confidence {confidence_score * 100:.2f}%")
        record_attendance(class_name)
        last_recognized_name = class_name
        last_recognition_time = current_time

    # Display result
    cv2.putText(image, f"Detected: {class_name} ({confidence_score * 100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Attendance System", image)

    # Break loop with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
conn.close()
print("Program ended. Database connection closed.")
