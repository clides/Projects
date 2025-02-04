import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model


# Load the model weights
def intializePredictionModel():
    model_weights_path = os.path.join(os.path.dirname(__file__), 'models', 'DigitDetectionCNN.h5')
    model = load_model(model_weights_path)
    return model

########### 1. Preprocessing the image
def preProcess(img):
    """
    Preprocess an image to prepare it for feature extraction.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The preprocessed image.
    """
    # Convert the image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Add a Gaussian blur to reduce noise and smoothen the image
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # Add an adaptive threshold to enhance edges and features
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    
    return imgThreshold


########### 2. Finding the biggest contour to assure that the sudoku is in the image
def biggestContour(contours):
    """
    Finds the biggest contour in the image, which should be the sudoku puzzle.

    Parameters:
    contours (list): A list of contours in the image.

    Returns:
    tuple: A tuple containing the biggest contour and its area.
    """
    # initialize variables
    biggest = np.array([])
    max_area = 0
    # iterate through each contour and find the biggest one
    for i in contours:
        # calculate the area of the contour
        area = cv2.contourArea(i)
        
        # check if the area found is just noise because it is so small
        if area > 50:
            # calculate the perimeter of the contour
            perimeter = cv2.arcLength(i, True)
            # approximate the contour to a polygon
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            # check if the area of the contour is greater than the current max area
            # and if the number of vertices in the polygon is 4
            if area > max_area and len(approx) == 4:
                # update the biggest contour and its area
                corners = approx
                max_area = area
    return corners, max_area


########### 3. Reorder the corners of the sudoku puzzle so it works with cv2.warpPerspective
def reorder(myPoints):
    """
    Reorders the points of a contour so that it works with cv2.warpPerspective.

    Parameters:
    myPoints (numpy.ndarray): A 2D array of points in the contour.

    Returns:
    numpy.ndarray: A 3D array with the points reordered.
    """
    
    # Reshape the points array to a 2D array
    myPoints = myPoints.reshape((4, 2))
    
    # Create a new array to store the reordered points
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)

    # Calculate the sum of x and y coordinates for each point
    add = myPoints.sum(1)
    
    # Find the point with the smallest sum (top-left corner)
    myPointsNew[0] = myPoints[np.argmin(add)]
    
    # Find the point with the largest sum (bottom-right corner)
    myPointsNew[3] = myPoints[np.argmax(add)]
    
    # Calculate the difference between x and y coordinates for each point
    diff = np.diff(myPoints, axis=1)
    
    # Find the point with the smallest difference (top-right corner)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    
    # Find the point with the largest difference (bottom-left corner)
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    return myPointsNew


########### 4. Split the image into 81 boxes
def splitBoxes(img):
    """
    Splits the given image into 81 smaller boxes (9x9 grid).

    Parameters:
    img (numpy.ndarray): The input image to be split.

    Returns:
    list: A list containing 81 smaller images (boxes).
    """
    # Split the image into 9 horizontal strips
    rows = np.vsplit(img, 9)
    boxes = []
    # Iterate over each strip
    for r in rows:
        # Split each strip into 9 vertical strips (boxes)
        cols = np.hsplit(r, 9)
        # Add each box to the list
        for box in cols:
            boxes.append(box)
    return boxes


########### 5. Get predictions on all the boxes
def getPrediction(boxes, model):
    """
    Gets the predictions for a list of boxes using the given model.

    Parameters:
    boxes (list): A list of images (boxes) to be processed.
    model (keras.Model): The model to be used for prediction.

    Returns:
    list: A list of the predicted class indices for each box.
    """
    result = []
    for image in boxes:
        # Preprocess each box so it works with the model
        img = np.asarray(image)
        # Remove the border of 4 pixels from the box
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        # Resize the box to 32x32 pixels
        img = cv2.resize(img, (32, 32))
        # Normalize the pixel values to be between 0 and 1
        img = img / 255.0
        # Reshape the box to be compatible with the model
        img = img.reshape(32, 32, 1)
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 32, 32, 1)
        
        # Get the prediction
        predictions = model.predict(img)  # Get the predicted class probabilities
        classIndex = np.argmax(predictions, axis=-1)  # Get the class index of the highest probability
        probabilityValue = np.max(predictions)  # Get the maximum probability
        
        # Save the result if the confidence is high enough
        if probabilityValue > 0.8:
            result.append(classIndex[0])  # Append the predicted class index
        else:
            result.append(0)  # If confidence is low, append 0 (unknown class)
    
    return result


########### 5. Display the numbers on the page
def displayNumbers(img, numbers, color = (0, 255, 0)):
    """
    Displays the given numbers on the given image.

    Parameters:
    img (numpy.ndarray): The image to draw the numbers on.
    numbers (list): A list of numbers (0-9) to be displayed on the image.
    color (tuple): The color of the text (BGR format). Defaults to green.

    Returns:
    numpy.ndarray: The image with the numbers drawn on it.
    """
    # Calculate the width and height of each cell in the Sudoku grid
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    fontScale = min(secW, secH) * 0.03
    
    # Loop over each cell in the grid in [y, x] format
    for x in range(0, 9):
        for y in range(0, 9):
            # If the number in the current cell is not 0, draw it on the image
            if numbers[(y * 9) + x] != 0:
                cv2.putText(
                    img, 
                    str(numbers[(y * 9) + x]), 
                    (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                    fontScale=fontScale,  # Adjust this value for different font sizes
                    color=color, 
                    thickness=2,  # Thickness of the text
                    lineType=cv2.LINE_AA  # Anti-aliased line
                )
    return img

def overlaySolutions(threshold, imgInvWarpColored, img):
    # Convert to BGRA to handle transparency
    imgInvWarpColored = cv2.cvtColor(imgInvWarpColored, cv2.COLOR_BGR2BGRA)

    # Create a mask for pixels close to black and set them to fully transparent
    black_pixels = np.all(imgInvWarpColored[:, :, :3] <= threshold, axis=2)
    imgInvWarpColored[black_pixels] = [0, 0, 0, 0]  # Set BGRA to [0, 0, 0, 0]
    imgInvWarpColored[~black_pixels, 3] = 255  # Set alpha to 255 for non-black pixels

    # Extract the alpha channel (transparency) from imgInvWarpColored
    alpha_channel = imgInvWarpColored[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
    alpha_channel = np.expand_dims(alpha_channel, axis=-1)  # Make it (height, width, 1)

    # Apply Gaussian blur to smooth out the sharp edges of the alpha channel
    alpha_channel_blurred = cv2.GaussianBlur(alpha_channel, (5, 5), 0)

    # Ensure alpha_channel_3d is 3-dimensional
    if alpha_channel_blurred.ndim == 2:
        alpha_channel_3d = np.expand_dims(alpha_channel_blurred, axis=-1)  # Add a third dimension
    else:
        alpha_channel_3d = alpha_channel_blurred

    # Repeat the alpha channel along the third axis to match the RGB shape for blending
    alpha_channel_3d = np.repeat(alpha_channel_3d, 3, axis=-1)

    # Extract the RGB channels from imgInvWarpColored
    overlay_rgb = imgInvWarpColored[:, :, :3]

    # Extract the RGB channels from the original image (img)
    img_rgb = img[:, :, :3]

    # Resize overlay_rgb and alpha_channel_3d to match the size of img_rgb
    overlay_rgb = cv2.resize(overlay_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
    alpha_channel_3d = cv2.resize(alpha_channel_3d, (img_rgb.shape[1], img_rgb.shape[0]))

    # Ensure alpha_channel_3d has the same shape as img_rgb and overlay_rgb
    if alpha_channel_3d.shape != img_rgb.shape:
        alpha_channel_3d = np.repeat(alpha_channel_3d[:, :, 0:1], 3, axis=-1)

    # Apply the alpha blending formula for smoother integration
    img_combined = cv2.convertScaleAbs(img_rgb * (1 - alpha_channel_3d) + overlay_rgb * alpha_channel_3d)

    # Add the blurred alpha channel to the combined image to maintain transparency
    img_final = cv2.merge([img_combined, (alpha_channel_3d[:, :, 0] * 255).astype(np.uint8)])
    return img_final