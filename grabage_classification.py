!pip install -q kaggle gradio tensorflow matplotlib numpy pillow

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mostafaabla/garbage-classification
!unzip -q garbage-classification.zip -d waste_dataset

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

Define dataset paths

dataset_path = '/content/waste_dataset/garbage_classification'
train_dir = '/content/waste_dataset/train'
val_dir = '/content/waste_dataset/val'

Create train and validation directories

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

Get class names dynamically

classes = [cls for cls in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cls))]

Create class directories inside train and val

for cls in classes:
os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

Split data (70% train, 30% val)

split_ratio = 0.7
for cls in classes:
class_path = os.path.join(dataset_path, cls)
images = os.listdir(class_path)

if len(images) < 2:  # Ensure at least one image in both train and val  
    continue  

random.shuffle(images)  
split_index = int(len(images) * split_ratio)  

for img in images[:split_index]:  
    shutil.move(os.path.join(class_path, img), os.path.join(train_dir, cls, img))  
for img in images[split_index:]:  
    shutil.move(os.path.join(class_path, img), os.path.join(val_dir, cls, img))

Image data generators

train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

Create data generators

train_generator = train_datagen.flow_from_directory(
train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

Get class names

class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)

Save class names for later use in prediction

with open('/content/class_names.txt', 'w') as f:
f.write("\n".join(class_names))

Load MobileNetV2 as base model

base_model = MobileNetV2(
input_shape=(224, 224, 3),
include_top=False,
weights='imagenet'
)

Freeze base model

base_model.trainable = False

Build the model

model = models.Sequential([
base_model,
layers.GlobalAveragePooling2D(),
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(len(class_names), activation='softmax')
])

Compile model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Train model

history = model.fit(
train_generator,
steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
validation_data=val_generator,
validation_steps=max(1, val_generator.samples // val_generator.batch_size),
epochs=10  # Change epochs if needed
)

Save trained model

model.save('/content/waste_classification_model.h5')

!pip install gradio --upgrade

import gradio as gr
import numpy as np
from PIL import Image

Load trained model

model = tf.keras.models.load_model('/content/waste_classification_model.h5')

Load class names

with open('/content/class_names.txt', 'r') as f:
class_names = f.read().splitlines()

Define prediction function

def predict_image(img):
if img is None:
return "Error: No image received. Please upload an image."

img = img.resize((224, 224))  # Resize to model input size  
img_array = image.img_to_array(img)  
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize  

# Predict  
predictions = model.predict(img_array)  
class_index = np.argmax(predictions)  
confidence = np.max(predictions)  

return f"Predicted: {class_names[class_index]} (Confidence: {confidence:.2%})"

Create GUI

interface = gr.Interface(
fn=predict_image,
inputs=gr.Image(type="pil"),
outputs="text",
title="Waste Classification",
description="Upload an image of waste to classify it into categories."
)

Launch GUI

interface.launch(share=True)
