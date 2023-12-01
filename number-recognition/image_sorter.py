### IMPORTS
import os, shutil, glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def split_data(parent_folder, csv_path, split_val=0.1, split_test=0.1, NOTMOVED = False, sideline=False, endzone = False):
  path_training_images = os.path.join(parent_folder, 'train')
  path_val_images = os.path.join(parent_folder, 'val')
  path_test_images = os.path.join(parent_folder, 'test')
  path_train_sideline = os.path.join(parent_folder,"sideline-train")
  path_val_sideline = os.path.join(parent_folder,"sideline-val")
  path_test_sideline = os.path.join(parent_folder,"sideline-test")
  path_train_endzone = os.path.join(parent_folder,"endzone-train")
  path_val_endzone = os.path.join(parent_folder,"endzone-val")
  path_test_endzone = os.path.join(parent_folder,"endzone-test")

  if sideline:
    df = pd.read_csv(csv_path, index_col='filename')
    images_paths = glob.glob(os.path.join(parent_folder, 'train_player_numbers', '*.png'))

    x_train, x_temp = train_test_split(images_paths, test_size=split_val + split_test, random_state=1254)
    x_val, x_test = train_test_split(x_temp, test_size=split_test / (split_val + split_test), random_state=1254)

    for path in [path_train_sideline, path_val_sideline, path_test_sideline]:
        if not os.path.isdir(path):
            os.makedirs(path)

    for (set_paths, path) in [(x_train, path_train_sideline), (x_val, path_val_sideline), (x_test, path_test_sideline)]:
        for image_path in set_paths:
            filename = os.path.basename(image_path)
            label = df.loc[filename, "label"]
            if isinstance(label, np.int64) and "Sideline" in filename:
                if label != 0:
                    path_to_folder = os.path.join(path, str(label))
                    if not os.path.isdir(path_to_folder):
                        os.makedirs(path_to_folder)
                    shutil.copy(image_path, path_to_folder)

  if endzone:                
    df = pd.read_csv(csv_path, index_col='filename')
    images_paths = glob.glob(os.path.join(parent_folder, 'train_player_numbers', '*.png'))

    x_train, x_temp = train_test_split(images_paths, test_size=split_val + split_test, random_state=1254)
    x_val, x_test = train_test_split(x_temp, test_size=split_test / (split_val + split_test), random_state=1254)

    for path in [path_train_endzone, path_val_endzone, path_test_endzone]:
        if not os.path.isdir(path):
            os.makedirs(path)

    for (set_paths, path) in [(x_train, path_train_endzone), (x_val, path_val_endzone), (x_test, path_test_endzone)]:
        for image_path in set_paths:
            filename = os.path.basename(image_path)
            label = df.loc[filename, "label"]
            if isinstance(label, np.int64) and "Endzone" in filename:
                if label != 0:
                    path_to_folder = os.path.join(path, str(label))
                    if not os.path.isdir(path_to_folder):
                        os.makedirs(path_to_folder)
                    shutil.copy(image_path, path_to_folder)

  if NOTMOVED:
    df = pd.read_csv(csv_path, index_col='filename')
    images_paths = glob.glob(os.path.join(parent_folder, 'train_player_numbers', '*.png'))

    x_train, x_temp = train_test_split(images_paths, test_size=split_val+split_test, random_state=1254)
    x_val, x_test = train_test_split(x_temp, test_size=split_test/(split_val+split_test), random_state=1254)

    for path in [path_training_images, path_val_images, path_test_images]:
      if not os.path.isdir(path):
        os.makedirs(path)

    for (set_paths, path) in [(x_train, path_training_images), (x_val, path_val_images), (x_test, path_test_images)]:
      for image_path in set_paths:
        filename = os.path.basename(image_path)
        label = df.loc[filename, "label"]
        if isinstance(label, np.int64):
          if label!=0:
            path_to_folder = os.path.join(path, str(label))
            if not os.path.isdir(path_to_folder):
              os.makedirs(path_to_folder)
            shutil.copy(image_path, path_to_folder)

def data_generators(batch_size,sideline=True, endzone=True):
  if sideline and endzone:
    train_data_path=os.path.join(os.getcwd(),"number-recognition","train")
    val_data_path=os.path.join(os.getcwd(),"number-recognition","val")
    test_data_path=os.path.join(os.getcwd(),"number-recognition","test")
  elif not sideline:
    train_data_path=os.path.join(os.getcwd(),"number-recognition","endzone-train")
    val_data_path=os.path.join(os.getcwd(),"number-recognition","endzone-val")
    test_data_path=os.path.join(os.getcwd(),"number-recognition","endzone-test")
  else:
    train_data_path=os.path.join(os.getcwd(),"number-recognition","endzone-train")
    val_data_path=os.path.join(os.getcwd(),"number-recognition","endzone-val")
    test_data_path=os.path.join(os.getcwd(),"number-recognition","endzone-test")
     
  preprocessor = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,     
    #width_shift_range=0.1,  
    #height_shift_range=0.1, 
    #shear_range=0.1,   
    zoom_range=0.2
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