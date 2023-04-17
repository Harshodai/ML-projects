#!/usr/bin/env python
# coding: utf-8

# In[9]:


pip install tensorflow


# In[21]:


pip install keras


# In[10]:


'''
image data generator is used like as feature scaling and not invloving colors from range 0 to 255 
and this will be like from range 0 to 1 and the image can squeezed or rotated or by any forms 
'''
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[11]:


tf.__version__


# # Preprocessing the data
# 

# preprocessing the training data set

# # Image data generator is used like as feature scaling and not involving colors from range 0 to 255 \nand this will be like from range 0 to 1 and the image can squeezed or rotated or by any forms

# In[12]:


'''
rescale - rescaling the values, can use 1.0 or 1.
shear range - rotating the image in counter clock wise direction it will be float 0.2 means 20% of the angle will be tilted
horizontal_flip - flipping the image horizontally
zoom range - zooming the image
target_size - resizing the dimensions of an image to 64*64 i.e height and breadth will be 64 * 64
class mode - "categorical" will be 2D one-hot encoded labels,
"binary" will be 1D binary labels,
"sparse" will be 1D integer labels,
"input" will be images identical to input images (mainly used to work with autoencoders).
If None, no labels are returned (the generator will only yield batches of image data, 
which is useful to use with model.predict()). Please note that in case of class_mode None, 
the data still needs to reside in a subdirectory of directory for it to work correctly.
'''
train_data = ImageDataGenerator(rescale = 1.0/255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                                horizontal_flip = True)
train_dataset = train_data.flow_from_directory('datasets/dogs_cats/training_set',
                                              target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'binary')


# preprocessing the testing dataset

# In[13]:


test_data = ImageDataGenerator(rescale = 1.0/255)
test_dataset = test_data.flow_from_directory('datasets/dogs_cats/test_set',
                                              target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'binary')


# # Initialising the CNN

# In[14]:


# adding a sequential neural network that means adding neurons in order wise one after the other 
cnn = tf.keras.models.Sequential()


# # Adding Convolution Layers to the neural network

# In[15]:


# to specify the characteristics of the first layer of the neural network
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))
#here we use 3 for 3 colors - red, green, blue or if we use 1 - then only black and white are used
# kernel size refers to the size of feature detectors as they are in square matrices 


# # Adding Pooling Layer

# In[16]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
# we use 2*2 square matrices for overlaying on the convolution layer and for pooling the maximum pixel size
# strides - matrix that is overlaid upon the feature detector will move by 2 cells with each iteration. 
''' if strides=2, the window will move 2 pixels to the right after each operation until 
it reaches the right edge of the input data,
at which point it will move down 2 pixels and start again from the left edge. 
This process continues until the entire input data has been covered by the sliding window. '''
#Note that since our pooling matrix is a 2x2 matrix, 
#this means that our pooling algorithm will not analyze any of the same pixels twice.


# # Adding another layer 
# ##Adding additional convolutional and pooling layers to a convolutional neural network can help improve its ability to recognize complex patterns in the input data. Each convolutional layer applies a set of filters to the input data to produce a set of feature maps that represent the presence and location of specific features in the input. By stacking multiple convolutional layers, the network can learn to recognize increasingly complex and abstract features. Pooling layers are used to downsample the feature maps and reduce their spatial dimensions, which can help reduce the computational cost of processing the data and improve the ability of the network to recognize patterns that are invariant to small translations.

# In[17]:


# for better prediction adding an other convolution network without adding the input_shape
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[18]:


cnn.add(tf.keras.layers.Flatten())


# In[19]:


# standalone layer
# dense is fully connected layer meaning output of the previous neurons are fed to the input of the present neurons
# we use 128 neurons 
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# # output layer

# In[20]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# # Compiling the Model

# In[21]:


# Adam Optimizer updates network weights iteratively based on training data in these applications.
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training the data in training set and testing on test_set

# In[22]:


cnn.fit(x = train_dataset, validation_data = test_dataset, epochs = 25)


# In[30]:


import numpy as np
import keras.utils as image
test_image = image.load_img('datasets/dogs_cats/single_prediction/cat_dog.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # reshuffling the dimensions
result = cnn.predict(test_image)
'''It is a dictionary that maps class names to their corresponding integer labels. 
This property is used to access the mapping between class names and their integer labels,
which can be useful when interpreting the predictions of a trained model.'''
train_dataset.class_indices # calls a function
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'


# In[31]:


print(result)


# In[32]:


print(prediction)


# In[ ]:




