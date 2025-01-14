#  TASK=1    Face Recognition Python 


![Screenshot 2025-01-14 104543](https://github.com/user-attachments/assets/95eb663c-7164-4cdb-bb3e-daa6bfa3e73c)


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
- 

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

