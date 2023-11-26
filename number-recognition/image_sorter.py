### IMPORTS
import os, shutil, glob
from sklearn.model_selection import train_test_split

def split_data(path_to_images,path_training_images,path_val_images,path_test_images, split_val=0.1, split_test=0.1):
  folders = os.listdir(path_to_images)

  for folder in folders:

    full_path = os.path.join(path_to_images,folder)
    images_paths = glob.glob(os.path.join(full_path, '*.png'))
    x_train, x_temp = train_test_split(images_paths, test_size=split_val+split_test, random_state=1254)
    x_val, x_test = train_test_split(x_temp, test_size=split_test/(split_test+split_val), random_state=1354)

    for (set,path) in [(x_train,path_training_images),(x_val,path_val_images),(x_test,path_test_images)]:
      for x in set:
        path_to_folder = os.path.join(path, folder)
        if not os.path.isdir(path_to_folder):
          os.makedirs(path_to_folder)
        shutil.copy(x,path_to_folder)