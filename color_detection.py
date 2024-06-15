import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Define the path to the dataset
dataset_path = 'base_path'

# Define the image dimensions
img_width, img_height = 64, 64

# Function to load images
def load_images(dataset_path, img_width, img_height):
    images = []
    labels = []
    label_names = os.listdir(dataset_path)
    label_map = {label: idx for idx, label in enumerate(label_names)}

    for label in label_names:
        folder_path = os.path.join(dataset_path, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(label_map[label])

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(label_names))

    return images, labels, label_names

# Load images and labels
images, labels, label_names = load_images(dataset_path, img_width, img_height)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_names), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('color_classification_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Function to predict the color of a new image
def predict_color(image_path, model, label_names, img_width, img_height):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    label_idx = np.argmax(prediction)
    return label_names[label_idx]

# Example usage
new_image_path = 'path_of_new_image'
predicted_color = predict_color(new_image_path, model, label_names, img_width, img_height)
print(f'The predicted color is: {predicted_color}')
