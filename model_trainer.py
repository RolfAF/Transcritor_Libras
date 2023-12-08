import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame(pd.read_csv('normalized/main.txt', sep=",", header=None))
#data = pd.read_csv('normalized/b.txt', sep=",", header=None)
#df = pd.DataFrame(data)
#main = pd.concat([main,df],axis=1)
#df.append(df2)
X = df.drop([60], axis=1)
y = df[60]
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Define the model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(60,)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(22, activation='softmax')  # Assuming 22 categories
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming your data is in the form of numpy arrays
# X_train: (num_samples, 60) array for training data
# y_train: (num_samples,) array for corresponding labels (categories)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Make predictions on new data
predictions = model.predict(new_data)

# 'predictions' will contain the predicted probability distribution for each category
# You can use np.argmax(predictions, axis=1) to get the predicted category for each example

