### IMPORTS
import tensorflow as tf
import os, random
import numpy as np
from image_sorter import data_generators
from PIL import Image, ImageDraw, ImageFont

def nn(num_classes):
    inputs = tf.keras.layers.Input(shape=(64, 64, 3), name='input_1')
    #x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv2d_0')(inputs)
    #x = tf.keras.layers.MaxPooling2D(name='max_pooling2d_0')(x)
    #x = tf.keras.layers.BatchNormalization(name='batch_normalization_0')(x)

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
  ### SWITCHES
  TRAIN = False
  TEST = False
  PREDICT = True

  ### DATA GENERATION / PIPELINE
  batch_size = 64 
  train_data,val_data,test_data = data_generators(batch_size,os.path.join(os.getcwd(),"number-recognition","train"),os.path.join(os.getcwd(),"number-recognition","val"),os.path.join(os.getcwd(),"number-recognition","test"))
  nbr_classes = train_data.num_classes

  class_index_to_label = {v: k for k, v in train_data.class_indices.items()} # conversion for innacurate class numbering

  ### SAVING MODEL AND ENABLING EARLY STOPPING
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

  if TRAIN:
    ### CREATING TF MODEL
    num_epochs = 100
    model = nn(nbr_classes)
    initial_learning_rate = 0.0003
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
    model = tf.keras.models.load_model("./number-recognition/Models")
    model.summary()

    ### EVALUATE ACCURACY BASED ON VAL / TEST
    print("Evaluating validation set: ")
    model.evaluate(val_data)
    print("Evaluating test set: ")
    model.evaluate(test_data)

  if PREDICT:
    ### MAKING RANDOMIZED PREDICTIONS
    model = tf.keras.models.load_model("./number-recognition/Models")
    correct=0
    amt=0
    confidences=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    top_images_count = 100
    top_images_info = []
    for i in range(1,100):
      test_path = os.path.join(os.getcwd(),"number-recognition","test")
      folder_path = os.path.join(test_path, str(i))
      filenames = os.listdir(folder_path)
      if filenames:
        num_files_to_select = min(30, len(filenames))
        random_filenames = random.sample(filenames, num_files_to_select)
        amt+=len(random_filenames)
        for random_filename in random_filenames:
          img_for_prediction_path = os.path.join(os.getcwd(),"number-recognition","test",str(i),random_filename)
          prediction,class_predicted = predict(model, img_for_prediction_path)
          confidence = round(100*np.max(prediction),3)
          confind = int(round(100*np.max(prediction),3))//10
          if confind >= 0:
            confidences[int(round(100*np.max(prediction),3))//10][1]+=1
            if int(class_index_to_label[class_predicted])==i:
              confidences[int(round(100*np.max(prediction),3))//10][0]+=1
          # print(f"\nActual: {i}, Predicted: ,{class_index_to_label[class_predicted]}, Confidence: {confidence}%")
          if int(class_index_to_label[class_predicted])==i:
            correct+=1
          top_images_info.append({
            'filename': random_filename,
            'predicted_class': int(class_index_to_label[class_predicted]),
            'true_class': i,
            'confidence': confidence
          })
          # print(random_filename)

    ### ACCURACY EVALUATION
    print(f"\nAccuracy {correct}/{amt}")
    print(confidences)
    for i in range(len(confidences)):
      try: print(f"\nAccuracy w/ Confidence of {(i)*10}%: {confidences[i][0]}/{confidences[i][1]}, {round(100*confidences[i][0]/confidences[i][1],1)}%")
      except: print(f"No predictions with Confidence of {i*10}%")


    ### PRINT CONFIDENT PREDICTIONS
    top_images_info.sort(key=lambda x: x['confidence'], reverse=True)
    top_images_info = top_images_info[:top_images_count]

    collage_width = 1000
    collage_height = 1000
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
    draw = ImageDraw.Draw(collage)

    font = ImageFont.load_default()

    for idx, info in enumerate(top_images_info):
      img_path = os.path.join(os.getcwd(), "number-recognition", "test", str(info['true_class']), info['filename'])
      img = Image.open(img_path)
      
      # Calculate position for pasting the image
      row = idx // 10
      col = idx % 10
      paste_x = col * (collage_width // 10)
      paste_y = row * (collage_height // 10)
      
      # Convert RGB image to HSV
      img_hsv = img.convert('HSV')
      
      # Extract hue, saturation, and value channels
      h, s, v = img_hsv.split()
      
      # Set the desired hue (0.5 corresponds to green, 0.0 to red)
      hue_adjustment = 0.5 if info['predicted_class'] == info['true_class'] else 0.0
      h = h.point(lambda i: i * hue_adjustment)
      
      # Merge the channels back together
      img_hue_adjusted = Image.merge('HSV', (h, s, v)).convert('RGB')
      
      # Paste the hue-adjusted image onto the collage
      collage.paste(img_hue_adjusted, (paste_x, paste_y))
      
      text = f"Pred: {info['predicted_class']}"
      text_position = (paste_x + 30, paste_y + 80)
      draw.text(text_position, text, font=font, fill=(0, 0, 0))

    collage.save('top_images_collage.png')

    ### MAKING SPECIFIC PREDICTIONS
    
