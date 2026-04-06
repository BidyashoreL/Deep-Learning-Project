import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"

# LOAD DATA
X = np.load(os.path.join(BASE_DIR, "data", "vj_cnn_input.npy"))[:668, :692, :]

# IMPROVED TRAINING LABEL LOGIC
Y = np.where(
    (X[:,:,0] < 0.35) &   # low elevation
    (X[:,:,2] > 0.5) &    # built-up
    (X[:,:,3] > 0.4) &    # far from drainage
    (X[:,:,4] > 0.4),     # heavy rainfall
    1, 0
)

Y = Y.reshape(668, 692, 1)

# MODEL
model = models.Sequential([
    layers.Input(shape=(668, 692, 5)),

    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu', padding='same'),

    layers.UpSampling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),

    layers.UpSampling2D(),
    layers.Conv2D(1, 1, activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train = np.expand_dims(X, axis=0)
Y_train = np.expand_dims(Y, axis=0)

print("🚀 Training 5-channel flood model...")
model.fit(X_train, Y_train, epochs=25)

model.save(os.path.join(BASE_DIR, "model", "vj_flood_cnn.h5"))

print("✅ 5-channel model trained & saved")