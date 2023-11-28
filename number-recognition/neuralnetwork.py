### IMPORTS
import tensorflow as tf
import os, random
import numpy as np
from image_sorter import data_generators

def nn(num_classes):
    inputs = tf.keras.layers.Input(shape=(64, 64, 3), name='input_1')
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d')(inputs)
    x = tf.keras.layers.MaxPooling2D(name='max_pooling2d')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization')(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_1')(x)
    x = tf.keras.layers.MaxPooling2D(name='max_pooling2d_1')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_1')(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_2')(x)
    x = tf.keras.layers.MaxPooling2D(name='max_pooling2d_2')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_2')(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv2d_3')(x)
    x = tf.keras.layers.MaxPooling2D(name='max_pooling2d_3')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_3')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
    x = tf.keras.layers.Dense(256, activation='relu', name='dense')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_1')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='model')
    return model

def predict(model,image_path):
  image = tf.io.read_file(image_path) # read file in
  image = tf.image.decode_png(image, channels=3) # decode from png to matrix
  image = tf.image.convert_image_dtype(image,dtype=tf.float32) # convert rgb (0-255) from int to (0-1) float32
  image = tf.image.resize(image,[64,64]) # resize to 64x64(x3) for rgb
  image = tf.expand_dims(image,axis=0) # expands shape to (1,64,64,3)

  predictions = model.predict(image) # return list of probabilities
  class_predicted = np.argmax(predictions) # returns class (index in predictions) of max
  return predictions,class_predicted

if __name__=="__main__":
  print("starting main")
  ### SWITCHES
  TRAIN = True
  TEST = True
  PREDICT = True

  ### DATA GENERATION / PIPELINE
  batch_size = 64 
  train_data,val_data,test_data = data_generators(batch_size,os.path.join(os.getcwd(),"number-recognition","train"),os.path.join(os.getcwd(),"number-recognition","val"),os.path.join(os.getcwd(),"number-recognition","test"))
  nbr_classes = train_data.num_classes

  class_index_to_label = {v: k for k, v in train_data.class_indices.items()} # conversion for innacurate class numbering

  ### SAVING MODEL AND ENABLING EARLY STOPPING
  path_to_save_model = "./Models"
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

  if TRAIN:
    ### CREATING TF MODEL
    num_epochs = 100
    model = nn(nbr_classes)
    initial_learning_rate = 0.0002
    final_learning_rate = 0.00005
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=num_epochs,
        decay_rate=decay_rate,
        staircase=False 
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
    
    ### FIT MODEL TO DATA
    model.fit(
      train_data,
      epochs=num_epochs,
      batch_size=batch_size,
      validation_data=val_data,
      callbacks=[ckptsaver,early_stop]
    )

  if TEST:
    ### IMPORT MODEL
    model = tf.keras.models.load_model("./Models")
    model.summary()

    ### EVALUATE ACCURACY BASED ON VAL / TEST
    print("Evaluating validation set: ")
    model.evaluate(val_data)
    print("Evaluating test set: ")
    model.evaluate(test_data)

  if PREDICT:
    ### MAKING RANDOMIZED PREDICTIONS
    model = tf.keras.models.load_model("./Models")
    correct=0
    np.set_printoptions(precision=4, suppress=True)
    for i in range(1,100):
      test_path = os.path.join(os.getcwd(),"number-recognition","test")
      folder_path = os.path.join(test_path, str(i))
      filenames = os.listdir(folder_path)
      if filenames:
        random_filename = random.choice(filenames)
        img_for_prediction_path = os.path.join(os.getcwd(),"number-recognition","test",str(i),random_filename)
        prediction,class_predicted = predict(model, img_for_prediction_path)
        print(f"\nClass: {i}, Class Predicted: ,{class_index_to_label[class_predicted]}, with confidence of {100*round(np.max(prediction),3)}%")
        if int(class_index_to_label[class_predicted])==i:
          correct+=1
        print(prediction)
        print(random_filename)
    print(f"\nAccuracy {correct}/100")

    ### MAKING SPECIFIC PREDICTIONS
    
