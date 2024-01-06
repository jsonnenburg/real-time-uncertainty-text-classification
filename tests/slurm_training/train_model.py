from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

from tensorflow.keras.callbacks import TensorBoard

tensorboard_callback = TensorBoard(histogram_freq=1)

# Load dataset (e.g., MNIST)
(train_images, train_labels), _ = mnist.load_data()
train_images = train_images / 255.0

# Define model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_images.reshape(-1, 784), train_labels, epochs=5, callbacks=[tensorboard_callback])

# Save model and training history
model.save('trained_model.h5')
with open('training_history.txt', 'w') as f:
    f.write(str(history.history))
