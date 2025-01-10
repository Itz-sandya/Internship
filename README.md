# Internship 
Project Report: Real-Time Face Recognition Using Keras and OpenCV

1. Introduction

This project focuses on implementing a real-time face recognition system using a deep learning model trained with Google Teachable Machine and deployed using Keras and OpenCV. The system captures video from the webcam, processes the frames, and predicts the class of the detected face with confidence scores.

2. Objectives

To implement a face recognition system using a pre-trained model.

To integrate real-time video processing using OpenCV.

To display prediction results with confidence scores.

3. Technologies Used

Python

TensorFlow

Keras

OpenCV

NumPy

4. System Requirements

Python 3.x

TensorFlow 2.x

Keras

OpenCV

NumPy

Webcam

5. Methodology

5.1 Loading the Model and Labels

A pre-trained model (keras_model.h5) and label file (labels.txt) are loaded for prediction.

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

5.2 Camera Initialization

OpenCV's VideoCapture is used to access the webcam for real-time video capture.

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

5.3 Image Preprocessing

Captured frames are resized to (224, 224) pixels to match the model's input requirements and normalized.

image_resized = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
image_array = (np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3) / 127.5) - 1

5.4 Model Prediction

The model predicts the class of the processed image, and the result is displayed with a confidence score.

prediction = model.predict(image_array)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

5.5 Display Results

The prediction and confidence score are printed, and the webcam feed is displayed until the ESC key is pressed.

cv2.imshow("Webcam Image", image)

6. Results

The system successfully captures video input, processes each frame, and outputs the predicted class with a corresponding confidence score.

7. Challenges Faced

Camera initialization issues on different operating systems.

Model compatibility and GPU availability.

8. Future Enhancements

Integrate face detection to improve recognition accuracy.

Deploy the model in a user-friendly GUI application.

Optimize performance for faster predictions.

9. Conclusion

This project demonstrates a basic real-time face recognition system using a deep learning model and OpenCV. It provides a foundation for more advanced applications in security and identity verification.

10. References

TensorFlow Documentation

Keras Documentation

OpenCV Documentation

Google Teachable Machine

