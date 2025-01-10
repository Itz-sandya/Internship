import cv2
import numpy as np
import mysql.connector
from tensorflow.keras.models import load_model
import time

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",  # Update with your MySQL root password
    database="AttendanceSystem"
)
c = conn.cursor()

# Function to record attendance in the database
def record_attendance(name):
    try:
        c.execute("SELECT * FROM attendance WHERE name = %s", (name,))
        if not c.fetchone():
            c.execute("INSERT INTO attendance (name, status) VALUES (%s, %s)", (name, 'present'))
            conn.commit()  # Ensure commit
            print(f"Attendance marked for {name}")
        else:
            print(f"Attendance already marked for {name}")
    except Exception as e:
        print(f"Error inserting attendance: {e}")

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

# Variables to control output
last_recognized_name = None
confidence_threshold = 0.95
cooldown_time = 3  # Cooldown time in seconds before allowing a new recognition
last_recognition_time = time.time()  # Initialize the last recognition time

while True:
    # Grab the web camera's image
    ret, image = camera.read()
    if not ret:
        print("Failed to capture image. Check your webcam.")
        break

    # Resize the raw image into (224-height, 224-width) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_array = (image_array / 127.5) - 1

    # Predict the model
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Clean the label
    confidence_score = prediction[0][index]

    current_time = time.time()

    # If the confidence is greater than the threshold and sufficient time has passed
    if confidence_score >= confidence_threshold and class_name != last_recognized_name and current_time - last_recognition_time >= cooldown_time:
        print(f"Recognized as: {class_name} with confidence {confidence_score*100:.2f}%")
        record_attendance(class_name)  # Mark attendance in the database
        last_recognized_name = class_name  # Update the last recognized name
        last_recognition_time = current_time  # Reset the cooldown timer

    # Display the image and recognition on the webcam feed
    cv2.putText(image, f"Detected: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Attendance System", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the ESC key on your keyboard.
    if keyboard_input == 27:
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()