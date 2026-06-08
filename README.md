# Driver Drowsiness Detection System

A real-time Driver Drowsiness Detection System built using Python, OpenCV, and MediaPipe Face Mesh.

The system monitors the driver's eyes through a webcam and detects drowsiness based on eye closure. When the eyes remain closed for a certain period, a visual **ALERT** is displayed on the screen.

## Project Overview

Driver drowsiness is one of the major causes of road accidents. This project aims to detect signs of driver fatigue in real time using a webcam and computer vision techniques.

The system is developed using Python, OpenCV, and MediaPipe Face Mesh. It continuously captures video from the webcam and tracks facial landmarks around the eyes. Using these landmarks, the Eye Aspect Ratio (EAR) is calculated to determine whether the driver's eyes are open or closed.

When the eyes remain closed for a specific number of consecutive frames, the system identifies the condition as drowsiness. It then displays a visual alert on the screen and logs the event for monitoring purposes.

### Working Process

1. Capture live video from the webcam.
2. Detect the driver's face using MediaPipe Face Mesh.
3. Extract eye landmark coordinates.
4. Calculate the Eye Aspect Ratio (EAR).
5. Compare the EAR value with a predefined threshold.
6. Monitor eye closure duration across consecutive frames.
7. Display an alert when drowsiness is detected.
8. Record detection events in a CSV log file.

## Features

* Real-time face detection
* Eye tracking using MediaPipe Face Mesh
* Eye Aspect Ratio (EAR) based drowsiness detection
* Visual alert system
* FPS monitoring
* CSV event logging

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy

## Project Screenshot

Project Screenshot uploaded above.

## Author

**Ashish Mishra**
ECE, NIT Silchar
