import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import necessary Flask modules
from flask import Flask, request, jsonify
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.models import load_model
import tensorflow as tf
from flask_cors import CORS


from numpy.random import seed
seed(0)
tf.keras.utils.set_random_seed(
    0
)

# Define the Flask application
app = Flask(__name__)
CORS(app)

# Load the quantized models
def load_quantized_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# pruned_alexnet_model = load_quantized_model("Quantized_AlexNet_model.tflite")
# pruned_resnet_model = load_quantized_model("Quantized_resnet_model.tflite")
# pruned_senet_model = load_quantized_model("Quantized_senet_model.tflite")

pruned_alexnet_model = load_model("AlexNet_model.h5")
pruned_resnet_model  = load_model("resnet_model.h5")
pruned_senet_model = load_model("senet_model.h5")
meta_learner = load_model("meta_learner.h5")


# def extract_features(model, processed_image):
#     input_details = model.get_input_details()
#     output_details = model.get_output_details()
    
#     # Assuming the model expects a batch dimension
#     input_shape = input_details[0]['shape']
#     if len(input_shape) == 4:
#         # Add a batch dimension if it's missing
#         processed_image = np.expand_dims(processed_image, axis=0)
    
#     # Prepare input tensor
#     input_data = processed_image.astype(np.float32)
#     model.set_tensor(input_details[0]['index'], input_data)

#     # Perform inference
#     model.invoke()

#     # Get the output tensor
#     output_data = model.get_tensor(output_details[0]['index'])
    
#     return output_data


def extract_features(model, processed_image):
    features_train = model.predict(processed_image)
    #features_test = model.predict(X_test)
    return features_train #, features_test

# Define a route for image upload and processing
@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    if request.method == 'GET':
        return jsonify({'message': 'Enviar un request del tipo POST con una imagen para procesar.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Ensure the file is a valid image
    if file.filename == '':
        return jsonify({'error': 'No selected file'})


    if file:
        # Read the image
        image = imread(file)

        # Resize the image
        resized_image = resize(image, (150, 250), anti_aliasing=True, preserve_range=True)

        #resized_image = np.array(resized_image, dtype=np.uint8)
        resized_image  = resized_image.astype('float32')

        # Get the dimensions of the resized image
        resized_image /=255.0

        resized_image = tf.keras.preprocessing.image.img_to_array(resized_image)
        resized_image = tf.expand_dims(resized_image, axis=0)

        # Extract features for each model
        alexnet_features = extract_features(pruned_alexnet_model, resized_image)
        resnet_features = extract_features(pruned_resnet_model, resized_image)
        senet_features = extract_features(pruned_senet_model, resized_image)


        #  # Stack the features
        stacked_features = np.concatenate([alexnet_features, resnet_features, senet_features], axis=1)

        # Make prediction using the meta-learner
        prediction = meta_learner.predict(stacked_features)
        print(prediction)
        predicted_class = np.argmax(prediction)

        return jsonify({'predicted_class': int(predicted_class)})


# Define a simple test route
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Ruta de testeo.'})

# Define the application function for WSGI
def aplicacion(environ, start_response):
    # Call the Flask application
    response = app(environ, start_response)
    return response

# Entry point of the application
if __name__ == '__main__':
    app.run(debug=True)
