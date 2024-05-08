import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import pathlib

# Path to the directory containing the training and testing images
train_data_dir = pathlib.Path('images/train').with_suffix('')
valid_data_dir = pathlib.Path('images/valid').with_suffix('')

image_count = len(list(train_data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
dim = (150, 150)
channel = (3, )
input_shape = dim + channel
img_height = 150
img_width = 150

train_ds = tf.keras.utils.image_dataset_from_directory(train_data_dir, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size, class_names=None, shuffle=True)
val_ds = tf.keras.utils.image_dataset_from_directory(train_data_dir, validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size, class_names=None, shuffle=True)

class_names = train_ds.class_names
print(class_names)

# Build the CNN model
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(class_names), activation='sigmoid')  # Ubah jumlah kelas menjadi sesuai dengan jumlah kelas dalam dataset Anda
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_ds, steps_per_epoch=train_ds.cardinality().numpy() // batch_size, epochs=image_count, validation_data=val_ds, validation_steps=val_ds.cardinality().numpy() // batch_size)

# Save the model
model.save('data/cnn/model.h5')