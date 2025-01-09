import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

########################## Initializations
path = 'myData'
testRatio = 0.2
validationRatio = 0.2
##########################

images = []
classNumber = []
myList = os.listdir(path)
numberOfClasses = len(myList)

print("Total Number of Classes Detected:", numberOfClasses)
print("Importing Classes...")

# resize each image from the dataset and add it to the images list
for folder_num in range(numberOfClasses):
    myPictList = os.listdir(path + '/' + str(folder_num))
    
    for pict_path in myPictList:
        curImg = cv2.imread(path + '/' + str(folder_num) + '/' + pict_path)
        curImg = cv2.resize(curImg,(32, 32))
        
        images.append(curImg)
        classNumber.append(folder_num)
        
    print(folder_num, end="")
    
print("")

# Converting the images to np arrays
images = np.array(images)
classNumber = np.array(classNumber)

print(images.shape)
print(classNumber.shape)


#### Splitting the data
X_train, X_test, y_train, y_test = train_test_split(images, classNumber, test_size=testRatio, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validationRatio, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


def checking_num_ratio():
    """
    Checks the number of images per digit class in the training data.
    
    This function prints out the number of images per digit class and displays it on a bar graph.
    """
    
    # create a list to store the number of samples per class
    numOfSamples = []
    
    # loop through each class
    for x in range(numberOfClasses):
        # get the number of samples in the current class
        numSamplesInCurrClass = len(np.where(y_train == x)[0])
        
        # add it to the list
        numOfSamples.append(numSamplesInCurrClass)
        
    # print out the number of samples per class
    print(numOfSamples)
    
    # display the result on a graph
    plt.figure(figsize=(10, 5))
    plt.bar(range(numberOfClasses), numOfSamples)
    plt.title("Number of Images per Digit Class")
    plt.xlabel("Class ID")
    plt.ylabel("Number of Images")
    plt.show()
# checking_num_ratio()

# function that preprocess an image
def preprocessImg(img):
    """
    Preprocess the given image to make it suitable for training and prediction.
    
    The function will turn the image to grayscale, equalize the image to make the lighting distribution uniform,
    and normalize the image to make the pixel values between 0 and 1.
    
    Parameters:
    img (numpy.ndarray): The image to be preprocessed.
    
    Returns:
    numpy.ndarray: The preprocessed image.
    """
    # turn the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # equalize the image to make the lighting distribution uniform
    img = cv2.equalizeHist(img)
    
    # normalize the image to make the pixel values between 0 and 1
    img = img / 255
    
    return img

# preprocess all X data with map method
X_train = np.array(list(map(preprocessImg, X_train)))
X_test = np.array(list(map(preprocessImg, X_test)))
X_val = np.array(list(map(preprocessImg, X_val)))

# add depth so it work with CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

# modify images to make the dataset more generic
dataGen = ImageDataGenerator(width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             zoom_range=0.2, 
                             shear_range=0.1, 
                             rotation_range=10)
dataGen.fit(X_train)

# one hot encoding the y data
y_train = to_categorical(y_train, numberOfClasses)
y_test = to_categorical(y_test, numberOfClasses)
y_val = to_categorical(y_val, numberOfClasses)



#### Create model