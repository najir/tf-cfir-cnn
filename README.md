# tf-cfir-cnn
    Isaac Perks
    06-20-2023

# Description

Using the cifar dataset to create a Convolutional Neural Network to classify some common object from 

CIFAR contains 60,000 images of 32x32 pixels, split between 10 different classifications of
everyday objects. Provided from Keras dataset, following tensorflows tutorial on CNN's

- Import and clean the dataset itself
    - Seperate data between training and evaluation sets & labes/images
    - Refactor the pixel values to be between 0 and 1 to match the models range
    - Save a list/array of labels for future comparison
    - Created a image generator to alter our testing data and create a larger dataset
- Build the model architecture
    - 3 main filter layers with 3x3 sample sizes and ReLU activation
    - 2 pooling layers to reduces sample sizes and feature map size
    - flatten our filters, distribute through network and output between 10 classifications
- Build/Train/Output our model
    - compile model with adam, probabilty distrobution for output, accuracy metric
    - log our model being fit to our training dataset comparing with evaluation set
        - epochs set to 8 for mix of speed and accuracy
    - evaluate and save our final model using evaluation data
        - tLoss, tAcc variables contain verbose lv 2 data from our evaluation

