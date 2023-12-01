import sys, sklearn, os, time, random, PIL, pandas as pd, numpy as np, cv2 as cv, matplotlib.pyplot as plt, tensorflow as tf
from tqdm import tqdm; from datetime import datetime

image_folder_path = "number-recognition/"
dataset_df = pd.read_csv("number-recognition/train_player_numbers.csv")
dataset_df

dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)
dataset_df["filepath"] = [image_folder_path+row.filepath for idx, row in dataset_df.iterrows()]
dataset_df = dataset_df[dataset_df.video_frame.str.contains("Endzone")]

training_percentage = 0.8
training_item_count = int(len(dataset_df)*training_percentage)
validation_item_count = len(dataset_df)-int(len(dataset_df)*training_percentage)
training_df = dataset_df[:training_item_count]
validation_df = dataset_df[training_item_count:]

batch_size = 64
image_size = 64
input_shape = (image_size, image_size, 3)
dropout_rate = 0.4
classes_to_predict = sorted(training_df.label.unique())
class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced",classes=classes_to_predict, y=training_df["label"].values)
class_weights_dict = {i : class_weights[i] for i,label in enumerate(classes_to_predict)}

training_data = tf.data.Dataset.from_tensor_slices((training_df.filepath.values, training_df.label.values))
validation_data = tf.data.Dataset.from_tensor_slices((validation_df.filepath.values, validation_df.label.values))

def load_image_and_label_from_path(image_path, label):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

training_data = training_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
validation_data = validation_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)

training_data_batches = training_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
validation_data_batches = validation_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

data_augmentation_layers = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
    tf.keras.layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
    tf.keras.layers.experimental.preprocessing.RandomContrast((0.2,0.2))
  ]
)

image = PIL.Image.open(training_df.filepath.values[1])
plt.imshow(image)
plt.show()

image = tf.expand_dims(np.array(image), 0)
plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation_layers(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")

efficientnet = tf.keras.applications.EfficientNetB0(weights="imagenet", 
                              include_top=False, 
                              input_shape=input_shape, 
                              drop_connect_rate=dropout_rate)

print(classes_to_predict)
inputs = tf.keras.layers.Input(shape=input_shape)
augmented = data_augmentation_layers(inputs)
efficientnet = efficientnet(augmented)
pooling = tf.keras.layers.GlobalAveragePooling2D()(efficientnet)
dropout = tf.keras.layers.Dropout(dropout_rate)(pooling)
outputs = tf.keras.layers.Dense(len(classes_to_predict), activation="softmax")(dropout)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
model.summary()

epochs = 25
decay_steps = int(round(len(training_df)/batch_size))*epochs
cosine_decay = tf.keras.experimental.CosineDecay(initial_learning_rate=1e-3, decay_steps=decay_steps, alpha=0.3)

path_to_save_model = "./number-recognition/Models"
ckptsaver = tf.keras.callbacks.ModelCheckpoint(
  path_to_save_model,
  monitor="val_accuracy",
  mode="max",
  save_best_only=True,
  save_freq="epoch",
  verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
  monitor="val_accuracy",
  patience=10
)

callbacks = [ckptsaver,early_stop]

model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(cosine_decay), metrics=["accuracy"])

history = model.fit(training_data_batches,
                  epochs = epochs, 
                  validation_data=validation_data_batches,
                  class_weight=class_weights_dict,
                  callbacks=callbacks)