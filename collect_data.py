import os
import cv2
import numpy as np
import mediapipe as mp

ACTIONS = ["MERHABA", "EVET", "HAYIR"]
SEQUENCE_LENGTH = 30  # Her örnek için kaç frame
DATA_DIR = "data"

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

os.makedirs(DATA_DIR, exist_ok=True)
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_DIR, action), exist_ok=True)

def extract_keypoints(results):
    import numpy as np

    # Sağ el
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    # Sol el
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Pose (vücut – 33 landmark, x y z visibility)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    return np.concatenate([rh, lh, pose])  # (42*3 + 33*4 = 258 boyutlu vektör)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    for action in ACTIONS:
        print(f"=== {action} için veri toplanıyor ===")
        sample_idx = 0

        while True:
            sequence = []  # 30 framelik bir dizi

            print(f"Yeni örnek için hazır mısın? 'q' = çıkış, 's' = başla")
            # Kullanıcı onayı
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"{action} - 's': kayda basla, 'q': cikis",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Collect Data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
                if key == ord('s'):
                    break

            # SEQUENCE_LENGTH kadar frame kaydet
            for i in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                # Çizimi göster
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                          mp_holistic.POSE_CONNECTIONS)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                cv2.putText(frame, f"{action} | ornek: {sample_idx} | frame: {i+1}/{SEQUENCE_LENGTH}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Collect Data", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)

            # Eğer tam sırayı topladıysa kaydet
            if len(sequence) == SEQUENCE_LENGTH:
                sequence = np.array(sequence)
                save_path = os.path.join(DATA_DIR, action, f"{sample_idx}.npy")
                np.save(save_path, sequence)
                print(f"Kaydedildi: {save_path}")
                sample_idx += 1
            else:
                print("Yeterli frame alınamadı, bu örnek atlandı.")
