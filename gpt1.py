import tensorflow as tf
from tensorflow.keras import layers, models

# Assuming each array has 21 (x, y) coordinates for hand landmarks
num_landmarks = 21
num_poses = 27

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(num_landmarks, 2)),  # Flatten the 21 (x, y) coordinates
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_poses, activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming your data is in the form of numpy arrays
# X_train: (2700, 21, 2) array for training data
# y_train: (2700,) array for corresponding labels (pose indices)
# You would need to preprocess your data accordingly

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Make predictions on new data
predictions = model.predict(new_data)

# Get the predicted pose index for each example
predicted_poses = tf.argmax(predictions, axis=1).numpy()

# Now, 'predicted_poses' contains the predicted pose indices for your test data

