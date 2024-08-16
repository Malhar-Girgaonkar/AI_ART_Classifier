import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


#loading model
#model_computed = Load_trained_model()

def Load_trained_model():
    model_path = r"Fine_tuning_models\finetune_model_V1.keras"
    model_computed = load_model(model_path)
    return(model_computed)

def preprocess(image_path):
    img_height , img_width = 224 , 224
    image_path = image_path
    #print(image_path)
    image_loaded = image.load_img(image_path , target_size = (img_height , img_width))
    image_array = image.img_to_array(image_loaded)
    img_array = np.expand_dims(image_array, axis=0)
    img_array /= 255.0
    print(f"Preprocessed image shape: {img_array.shape}")
    return(img_array)

def Predictions(img_path , progress_callback=None):
    model_computed = Load_trained_model()
    #update progressbar to 1/4
    if progress_callback:
        progress_callback(25)
    #preprocess data
    img_array = preprocess(img_path)
    #update progressbar to 2/4
    if progress_callback:
        progress_callback(50)
    #predict with model_computed
    prediction = model_computed.predict(img_array)
    return(prediction)
    #update progressbar to 3/4
    if progress_callback:
        progress_callback(75)
    #set threshold
    threshold = 0.5
    #print(prediction)
    if prediction[0][0] > threshold:
        #print(f"predicted value = {prediction[0][0]} and class AI ")
        #update progressbar to 4/4
        if progress_callback:
            progress_callback(100)
        return("AI Generated art")
    else:
        #print(f"predicted value = {prediction[0][0]} and class Human ")
        #update progressbar to 4/4
        if progress_callback:
            progress_callback(100)
        return("Human generated art")


def selectimg():
    #get image location
    img_destination_path=r"App data\Images"
    #ask image selection from user
    img_source_path=filedialog.askopenfilename(initialdir="D:",title="Select Image",filetypes=(("All files","*.*"),("JPG files","*.jpg")))
    #if image source= image destination then img_path=imgsource
    if(img_source_path == os.path.join(img_destination_path, os.path.basename(img_source_path))):
        img_path=img_source_path
    else:
        #move image to locationApp data\Images
        img_path=shutil.copy(img_source_path, img_destination_path)
    return(img_path)

def clean():
    #delete all images in folder App data\Images
    # Iterate over all the files in the directory
    directory_path=r"App data\Images"
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                # Delete the file
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")