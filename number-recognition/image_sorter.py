### IMPORTS
import os, shutil, glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def split_data(parent_folder, csv_path, split_val=0.1, split_test=0.1):
  df = pd.read_csv(csv_path, index_col='filename')
  images_paths = glob.glob(os.path.join(parent_folder, 'number-recognition', 'train_player_numbers', '*.png'))

  x_train, x_temp = train_test_split(images_paths, test_size=split_val + split_test, random_state=1254)
  x_val, x_test = train_test_split(x_temp, test_size=split_test / (split_test + split_val), random_state=1354)

  path_training_images = os.path.join(parent_folder, 'number-recognition', 'train')
  path_val_images = os.path.join(parent_folder, 'number-recognition', 'val')
  path_test_images = os.path.join(parent_folder, 'number-recognition', 'test')

  for path in [path_training_images, path_val_images, path_test_images]:
    if not os.path.isdir(path):
      os.makedirs(path)

  for (set_paths, path) in [(x_train, path_training_images), (x_val, path_val_images), (x_test, path_test_images)]:
    for image_path in set_paths:
      filename = os.path.basename(image_path)
      label = df.loc[filename, "label"]
      if isinstance(label,np.int64):
        path_to_folder = os.path.join(path, str(label))
        if not os.path.isdir(path_to_folder):
            os.makedirs(path_to_folder)

        shutil.copy(image_path, path_to_folder)

def data_generators(batch_size,train_data_path,val_data_path,test_data_path):
  preprocessor = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    #rotation_range=15,     
    #width_shift_range=0.1,  
    #height_shift_range=0.1, 
    #shear_range=0.1,   
    zoom_range=0.1
  )
  test_preprocessor = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255.
  )
  train_generator = preprocessor.flow_from_directory(
    train_data_path,
    class_mode="categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size
  )  
  val_generator = test_preprocessor.flow_from_directory(
    val_data_path,
    class_mode="categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size
  )  
  test_generator = test_preprocessor.flow_from_directory(
    test_data_path,
    class_mode="categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size
  )
  return train_generator,val_generator,test_generator 

#split_data(
  #parent_folder=os.getcwd(),
  #csv_path="number-recognition/train_player_numbers.csv"
#)