import tensorflow as tf
from tensorflow.keras import layers, models

# Assuming each numerical array has 21 hand landmarks (x, y coordinates)
num_landmarks = 21
num_poses = 27

# Input layer
input_layer = layers.Input(shape=(num_landmarks, 2), name='input_pose')

# Siamese network with shared layers
shared_layer = layers.Dense(128, activation='relu')(input_layer)
shared_layer = layers.Dropout(0.5)(shared_layer)  # Optional dropout layer for regularization

# Create 27 branches, each for one pose
outputs = []
for _ in range(num_poses):
    branch = layers.Dense(64, activation='relu')(shared_layer)
    branch = layers.Dense(32, activation='relu')(branch)
    branch = layers.Dense(1, activation='sigmoid')(branch)
    outputs.append(branch)

# Create the model
model = models.Model(inputs=input_layer, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming your data is in the form of numpy arrays
# X_train: (2700, 21, 2) array for training data
# y_train: (2700, 27) array for corresponding labels (match percentages)

# Train the model
model.fit(X_train, [y_train[:, i] for i in range(num_poses)], epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, [y_test[:, i] for i in range(num_poses)])
print(f'Test accuracy: {test_acc}')

# Make predictions on new data
predictions = model.predict(new_data)

# 'predictions' will be a list of 27 arrays, each representing the match percentage for one pose

#In this example, the model has 27 branches, each producing a single output value between 0 and 1, representing the match percentage for one of the 27 poses. The model is trained using binary cross-entropy loss for each branch. The test accuracy is evaluated for each pose independently. Make sure your labels (y_train and y_test) are normalized between 0 and 1 to match the model's output format. If not, you may need to normalize them before training.


