import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os
from playsound import playsound
from threading import Thread

EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 20

LOG_FILE = "drowsiness_log.csv"
ALARM_FILE = "alarm.wav"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event"])

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Left Eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right Eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# FUNCTIONS
def play_alarm():
    try:
        playsound(ALARM_FILE)
    except:
        pass


def euclidean(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])

    if C == 0:
        return 0

    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_points(indices, landmarks, w, h):
    points = []

    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))

    return points
# CAMERA
cap = cv2.VideoCapture(0)

closed_frames = 0
alarm_on = False

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            left_eye = get_eye_points(
                LEFT_EYE,
                landmarks,
                w,
                h
            )

            right_eye = get_eye_points(
                RIGHT_EYE,
                landmarks,
                w,
                h
            )

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            # eye points
            for p in left_eye:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            for p in right_eye:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if ear < EAR_THRESHOLD:

                closed_frames += 1

                cv2.putText(
                    frame,
                    "Eyes Closed",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    
                )

                if closed_frames >= CONSEC_FRAMES:

                    cv2.putText(
                        frame,
                        "ALERT!",
                        (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5,
                        (0, 0, 255),
                        6
                    )

                    if not alarm_on:

                        alarm_on = True

                        Thread(
                            target=play_alarm,
                            daemon=True
                        ).start()

                        with open(
                            LOG_FILE,
                            "a",
                            newline=""
                        ) as f:

                            writer = csv.writer(f)

                            writer.writerow(
                                [
                                    time.strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    "ALERT!"
                                ]
                            )

            else:
                closed_frames = 0
                alarm_on = False

    # FPS
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2
    )

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
