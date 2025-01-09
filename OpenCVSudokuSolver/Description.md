Using OpenCV to preprocess the image -> finding contours -> finding the -> classify digits -> find solution -> overlay solutions

Key takeaways:
- **OpenCV**
    - Workflow of preprocessing the image (grayscale → gaussianblur → adaptive threshold → finding contours)
    - Using OpenCV to stack images for better display
    - Using findCountours function from cv2 to find all the contours of image
	    - Find biggest contour by calculating area of each one with cv2.contourArea and use cv2.approxPolyDP to check whether the contour have 4 corners (ensures that the sudoku is the image)
    - Using cv2.warpPerspective to flatten the biggest contour (first by computing a perspective transformation matrix)
    - Splitting the sudoku image into 81 boxes using np.vsplit and np.hsplit

- **Tensorflow**
    - Using tensorflow to train a CNN model to recognize digits and blank spaces and then loading the model