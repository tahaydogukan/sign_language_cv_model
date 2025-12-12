import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

ACTIONS = ["MERHABA", "EVET", "HAYIR"]
DATA_DIR = "data"
SEQUENCE_LENGTH = 30

# 1) X, y dizilerini hazırla
sequences = []
labels = []

for label_idx, action in enumerate(ACTIONS):
    action_dir = os.path.join(DATA_DIR, action)
    for file_name in os.listdir(action_dir):
        if file_name.endswith(".npy"):
            path = os.path.join(action_dir, file_name)
            seq = np.load(path) 
            sequences.append(seq)
            labels.append(label_idx)

X = np.array(sequences)            
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# 2) LSTM modeli
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, X.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(ACTIONS), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)
model.save("sign_lstm_model.h5")
print("Model kaydedildi: sign_lstm_model.h5")
