import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from collections import deque, Counter

ACTIONS = ["MERHABA", "EVET", "HAYIR"]
SEQUENCE_LENGTH = 30

# Stabilizasyon ayarları
PRED_BUFFER = 12
MIN_MAJORITY = 0.7
CONF_THRESHOLD = 0.7

# 1) Modeli yükle
model = load_model("sign_lstm_model.h5")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] 
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] 
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    return np.concatenate([rh, lh, pose])

cap = cv2.VideoCapture(0)
sequence = []
pred_buffer = deque(maxlen=PRED_BUFFER)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        text = "Tahmin yok"
        color = (0, 0, 255)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            preds = model.predict(input_data, verbose=0)[0]

            pred_class = int(np.argmax(preds))
            pred_label = ACTIONS[pred_class]
            confidence = float(preds[pred_class])

            if confidence >= CONF_THRESHOLD:
                pred_buffer.append(pred_label)

            if len(pred_buffer) >= max(5, PRED_BUFFER // 2):
                counts = Counter(pred_buffer)
                best_label, best_count = counts.most_common(1)[0]
                majority_ratio = best_count / len(pred_buffer)

                text = f"{best_label} (vote {majority_ratio:.2f})"
                color = (0, 255, 0) if majority_ratio >= MIN_MAJORITY else (0, 255, 255)
            else:
                text = f"{pred_label} ({confidence:.2f})"
                color = (0, 255, 255) if confidence > 0.4 else (0, 0, 255)

        cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, "Cikmak icin 'q'", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Realtime Sign LSTM (Silent)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
