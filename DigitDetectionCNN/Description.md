**Key takeaways**:
- Using sklearn to split data into training, testing, validation sets
- Using matplotlib to visualize data
- Using cv2 to preprocess image (image -> grayscale -> equalize -> normalize)
- Using custom model to modify images to make it more generic and to_categorical to one hot encode the y data
- Creating a **LuNet Model** from scratch using convolutional layers, pooling layers, dropout layers, flatten layers, dense layers, softmax activation function, and relu activation function
    - <u>Pooling Layers</u>:
        - Used to reduce the spatial dimensions of feature maps while retaining the most important information
        - Reduces Overfitting: By reducing the dimensions of feature maps.
        - Improves Generalization: Ensures the model captures dominant features regardless of small variations in input.
        - Speeds Up Training: By reducing the number of parameters and computations.
    - <u>Dropout Layers</u>:
        - Works by randomly "dropping out" (i.e., setting to zero) a fraction of neurons during training -> prevents the network from relying too heavily on specific neurons, thereby encouraging it to learn more robust and generalized patterns.
        - Reduces Overfitting: Prevents the model from learning overly complex patterns that may not generalize to unseen data.
        - Encourages Robustness: Forces the network to learn more distributed representations by not relying too heavily on specific neurons.
        - Improves Generalization: Helps the network perform better on test data.
    - <u>Flatten Layers</u>:
        - Reshapes multidimensional input data (e.g., images represented as matrices or tensors) into a one-dimensional array (vector)
        - Required step before connecting convolutional or pooling layers to dense (fully connected) layers, which only accept 1D inputs
    - <u>Dense Layers</u>:
        - Connect every neuron (unit) from the previous layer to every neuron in the current layer, enabling complex transformations and learning of high-level features
- Training the model and plotting the loss and accuracy with matplotlib
- Loading the trained model and making predictions

**Summary of Workflow**
1. Importing all the image data from the folder and converting them to numpy arrays
2. Splitting the data into training, testing, and validation sets
3. Preprocessing each image (image -> grayscale -> equalize -> normalize -> adding depth so it work with CNN -> augmenting it); one-hot encoding the labels (y values)
4. Create model based on LuNet Model (with convolutional layers, dense layers, dropout layers, flatten layers, pooling layers, and non-linear activation functions)
5. Training the model
6. Plotting the loss and accuracy
7. Evaluating the model