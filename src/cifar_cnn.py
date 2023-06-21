import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Import our dataset from keras
(trainImages, trainLabels), (evalImages, evalLabels) = datasets.cifar10.load_data()

#Resizing our data values to be between 0 and 1, instead of 0 and 255 to fit our model better
trainImages /= 255.0
trainLabels /= 255.0

#Create a generator to alter our data and expand the dataset
dataGen = ImageDataGenerator(
    rotation_range     = 40,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    shear_range        = 0.2,
    zoom_range         = 0.2,
    horizontal_flip    = True,
    fill_mode          = 'nearest'
)
#!This exactly implementation is untested atm
(trainImages, trainLabels) = dataGen.flow(trainImages, trainLabels, batch_size=8, subset='training')

#Creating a list/array of label names to use for future comparisons
classNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

#Building the architecture of our CNN model
cnnModel = model.sequential()
cnnModel.add(layers.conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
cnnModel.add(layers.MaxPooling2D((2,2)))
cnnModel.add(layers.conv2D(64, (3,3), activation='relu'))
cnnModel.add(layers.MaxPooling2D((2,2)))
cnnModel.add(layers.conv2D(64, (3,3), activation='relu'))

#Set up the models input, output layers for final filters
cnnModel.add(layers.flatten())
cnnModel.add(layers.Dense(64, activation='relu'))
cnnModel.add(layers.Dense(10))

#Compile, fit, test, evaluate our model and saves the information for viewing and testing
cnnModel.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric = ['accuracy']
)
log = cnnModel.fit(trainImages, trainLabels, epochs=8, validation_data=(evalImages, evalLabels))
tLoss, tAcc = cnnModel.evaluate(evalImages, evalLabels, verbose=2)

