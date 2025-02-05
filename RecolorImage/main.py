import numpy as np
import cv2
import os

# https://github.com/richzhang/colorization/tree/caffe/colorization/models
# Download the caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

base_path = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(base_path, "models/colorization_deploy_v2.prototxt")
model_path = os.path.join(base_path, "models/colorization_release_v2.caffemodel")
kernal_path = os.path.join(base_path, 'models/pts_in_hull.npy')
image_path = os.path.join(base_path, 'Resources/3.jpg')

# load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernal_path)

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

bw_image = cv2.imread(image_path)
normalized = bw_image.astype(np.float32) / 255 # normalize the image
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

# resize the image to match the input size of the model
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0] # extract the Lightness channel
L -= 50

# feed the lightness channel to the model
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward() # perform forward pass to get ab channels
ab = np.squeeze(ab)
ab = ab.transpose((1, 2, 0))

ab_resized = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
L = cv2.split(lab)[0]

# combine the lightness and ab channels
colorized = np.concatenate((L[:, :, np.newaxis], ab_resized), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = (colorized * 255).astype(np.uint8)

cv2.imshow('black and white', bw_image)
cv2.imwrite('bw.jpg', bw_image)
cv2.imshow('colorized', colorized)
cv2.imwrite('colorized.jpg', colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()