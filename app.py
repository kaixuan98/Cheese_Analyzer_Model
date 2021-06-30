from flask import Flask, jsonify, request 
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import tensorflow as tf
import numpy as np
from PIL import Image
app = Flask(__name__)

test_input = '/testing_data/bad_cheese.jpeg'
test_target = 0 
IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
# loading model here 
 # TODO: new_cheese model 
new_model = tf.keras.models.load_model('cheese_model/cheese_model.h5')

# def get_prediction(image_bytes):


def process_img(img):
    img = conv_img(img)
    img = scale_img(img)
    return img

def conv_img(img):
    return img_to_array(load_img(img, target_size=IMG_DIM))

def scale_img(img):
    img_scaled = img.astype("float32")
    img_scaled /= 255 

    return img_scaled

def get_prediction(image):
    tensor = np.array(conv_img(image))
    scaled = np.reshape(scale_img(tensor), (1, 300, 300, 3 ))
    print(scaled.shape)
    predictions = new_model.predict(scaled)
    output = np.argmax(predictions[0])
    # _, y_hat = outputs.max(1)
    print(f"Output:{output}")
    print(f"Predictions:{predictions}")
    # TODO : if statements output 
        
    # predicted_idx = str(y_hat.item())
    # TODO: returns are class id and class name
    return None, None 

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # TODO: uncommennt 52, pass file into get_prediction, 57
    # if request.method == 'POST':
    # file = request.files['file']
    # file = open("/testing_data/bad_cheese.jpeg", "rb")
    # img_bytes = file.read()
    class_id, class_name = get_prediction("testing_data/good_cheese.jpeg")
    print(class_id, class_name)
    # return jsonify({'class_id': "class_id", 'class_name': "class_name"})
    return "Predicted Value is {}".format(class_name)