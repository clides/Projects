import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

########################## Initializations
path = 'myData'
testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32, 32, 3)
batch_size = 50
epochs_value = 10
steps_per_epoch = 2000
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
        curImg = cv2.resize(curImg,(imageDimensions[0], imageDimensions[1]))
        
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
    
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Equalize the image histogram
    img = cv2.equalizeHist(img)
    
    # Normalize the image to [0, 1] by dividing by 255
    img = img / 255.0
    
    return img

# preprocess all X data with map method
X_train = np.array(list(map(preprocessImg, X_train)))
X_test = np.array(list(map(preprocessImg, X_test)))
X_val = np.array(list(map(preprocessImg, X_val)))

# add depth so it work with CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

# augment image to make it more generic
dataGen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)

# one hot encoding the y data
y_train = to_categorical(y_train, numberOfClasses)
y_test = to_categorical(y_test, numberOfClasses)
y_val = to_categorical(y_val, numberOfClasses)


#### Create model based on LuNet Model
def myModel():
    """
    This function creates a model based on LuNet model.
    
    The model is a convolutional neural network (CNN) with 3 convolutional layers, 
    2 maxpooling layers, 2 dropout layers, and a flatten layer. The output layer is 
    a dense layer with the softmax activation function, which is used for multi-class 
    classification.
    
    Parameters:
    None
    
    Returns:
    keras.Model: The model created.
    """
    # defining parameters to be used in the model
    numberOfFilters = 60  # number of filters in the first 2 convolutional layers
    sizeOfFilter1 = (5, 5)  # size of the filters in the first 2 convolutional layers
    sizeOfFilter2 = (3, 3)  # size of the filters in the last convolutional layer
    sizeOfPool = (2, 2)  # size of the max pooling layers
    numberOfNodes = 500  # number of nodes in the dense layer
    
    model = Sequential()
    
    # first layer should use an Input layer instead of directly specifying input_shape in Conv2D
    model.add(Input(shape=(imageDimensions[0], imageDimensions[1], 1)))  # Input layer
    
    # first convolutional layer
    model.add(Conv2D(numberOfFilters, sizeOfFilter1, activation='relu'))
    # second convolutional layer
    model.add(Conv2D(numberOfFilters, sizeOfFilter1, activation='relu'))
    # first maxpooling layer
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    # third convolutional layer
    model.add(Conv2D(numberOfFilters//2, sizeOfFilter2, activation='relu'))
    # fourth convolutional layer
    model.add(Conv2D(numberOfFilters//2, sizeOfFilter2, activation='relu'))
    # second maxpooling layer
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    # first dropout layer
    model.add(Dropout(0.5))
    
    # flatten the output of the convolutional layers
    model.add(Flatten())
    # first dense layer
    model.add(Dense(numberOfNodes, activation='relu'))
    # second dropout layer
    model.add(Dropout(0.5))
    # output layer
    model.add(Dense(numberOfClasses, activation='softmax'))
    # compile the model
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

# train the model
history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size), 
    steps_per_epoch = steps_per_epoch,
    epochs = epochs_value, 
    validation_data = (X_val, y_val),
    shuffle = True
)

# Plotting the loss
def plot_loss(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.savefig("loss.png")
    plt.show()
    
# Plotting the accuracy
def plot_accuracy(history):
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig("accuracy.png")
    plt.show()
    
# Evaluating the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_loss(history)
plot_accuracy(history)

# save the model
model.save('DigitDetectionCNN.h5')