import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

tf.config.list_physical_devices('GPU')

# Load and preprocess the Fer2013 dataset
train_data_dir = 'data/train'
test_data_dir = 'data/test'
img_width, img_height = 75, 75
batch_size = 32

# Create data generators for training and validation with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Load the Inception-v3 model pre-trained on ImageNet data
base_model = InceptionV3(weights='imagenet', include_top=False)

# Fine-tune some of the top layers of the base model
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer with 1024 units
x = Dense(1024, activation='relu')(x)

# Add a dropout layer for regularization
x = tf.keras.layers.Dropout(0.5)(x)

# Add an output layer with softmax activation for emotion classification
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Print model summary
model.summary()

# Compile the model with a lower initial learning rate
initial_lr = 0.001
model.compile(optimizer=Adam(learning_rate=initial_lr), loss='categorical_crossentropy', metrics=['accuracy'])

# Adjust the learning rate schedule
initial_lr = 0.001
def lr_schedule(epoch):
    if epoch < 50:
        return initial_lr
    else:
        return initial_lr * 0.1

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model with data generators and learning rate scheduler
num_epochs = 100
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[lr_scheduler])

model_json=model.to_json()
with open("emotion_model_inception.json","w") as json_file:
    json_file.write(model_json)

# Save the trained model
model.save('emotion_detection_model.h5')
