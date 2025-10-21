import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load classifier
model = joblib.load("mudra_classifier_dual.pkl")

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(max_num_hands=2)

# Webcam
cap = cv2.VideoCapture(0)

def extract_landmarks_realtime(frame):
    """Return 126-dim landmarks for classifier"""
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand_coords = [np.zeros(63), np.zeros(63)]
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            landmarks = [coord for l in handLms.landmark for coord in (l.x, l.y, l.z)]
            if results.multi_handedness[idx].classification[0].label == "Left":
                hand_coords[0] = np.array(landmarks)
            else:
                hand_coords[1] = np.array(landmarks)
    return np.concatenate(hand_coords), results.multi_hand_landmarks

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, detected_hands = extract_landmarks_realtime(frame)

    # Draw hands
    if detected_hands:
        for handLms in detected_hands:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Predict mudra
    pred = model.predict([landmarks])
    cv2.putText(frame, f"Mudra: {pred[0]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Dual-Hand Mudra Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
