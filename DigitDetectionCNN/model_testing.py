import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('DigitDetectionCNN.h5')

def preprocessImg(img):
    """
    Preprocess an image to prepare it for feature extraction.

    The function resizes the image to 32x32 pixels, converts it to grayscale,
    applies histogram equalization to improve contrast, normalizes the pixel values
    to [0, 1] and reshapes the image to be suitable for feeding into the CNN.

    Parameters:
    img (numpy.ndarray): The input image to be preprocessed.

    Returns:
    numpy.ndarray: The preprocessed image.
    """
    # Convert the image to an array
    img = np.asanyarray(img)
    # Resize the image to 32x32 pixels
    img = cv2.resize(img, (32, 32))
    
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization to improve contrast
    img = cv2.equalizeHist(img)
    # Normalize the pixel values to [0, 1]
    img = img / 255.0
    
    # Reshape the image to be suitable for feeding into the CNN
    img = img.reshape(1, 32, 32, 1)
    return img  # Return the processed image

# Read and preprocess the image
img = cv2.imread("Resources/7.png")
img = preprocessImg(img)

# Make predictions using the model
predictions = model.predict(img)
print("The model's prediction is: " + str(np.argmax(predictions)))