#!/usr/bin/env python
# coding: utf-8

# In[1]:




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

 

from sklearn.preprocessing import LabelBinarizer,  StandardScaler 

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, roc_curve

# Tensorflow 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,MaxPool2D,MaxPooling2D, BatchNormalization, Conv2D, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical



import pandas as pd
import numpy as np


# # Looking at the data

# In[2]:


with np.load('train_data_label.npz') as data:
    train_data = data['train_data']
    train_label = data['train_label']


# In[3]:


with np.load('test_data_label.npz') as data:
    test_data = data['test_data']
    test_label = data['test_label']


# In[4]:


print(train_data.shape, train_data.dtype)


# In[5]:


print(train_data)
print(train_label)


# In[6]:


plt.figure(figsize = (10,10)) # Label Count
sns.set_style("darkgrid")
sns.countplot(train_label)


# In[7]:


np.unique(train_label, return_counts=True)


# In[8]:


np.unique(train_data)


# # Splitting the data
# Also reshaping it for CNN

# In[9]:


train = train_data
test = test_data


# In[10]:


train = train.astype('float32')
test = test.astype('float32')
train = train / 255.
test = test / 255.


# In[11]:


train_cnn = train.reshape(-1,28,28,1)
test_cnn = test.reshape(-1, 28,28, 1)
train.shape, test.shape


# In[12]:


y_train = train_label
y_test = test_label
print(y_train.shape)
# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)

# Display the change for category label using one-hot encoding
print('Original label:', y_train[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

label_as_binary = LabelBinarizer()
train_y_labels = label_as_binary.fit_transform(train_Y_one_hot)
test_y_labels = label_as_binary.fit_transform(test_Y_one_hot)
train_y_labels.shape


# In[13]:


X_train, X_val, y_train, y_val = train_test_split(train_cnn, train_y_labels, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2
X_train.shape, X_val.shape,y_train.shape, y_val.shape


# In[14]:


X_train


# In[15]:


f, ax = plt.subplots(2,5) 
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(X_train[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()  


# # Seed value

# In[16]:



seed_value= 22


# 1. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 2. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 3. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)


# # Data augmentation
# In order to avoid overfitting problem, we need to expand artificially our dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations.
# 
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.
# 
# By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.
# https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy

# In[17]:


# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


# # CNN
# also added dropout to counter overfitting (Thomas)

# In[18]:


batch_size = 35
epochs = 5
num_classes = 25

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(Dropout(0.3))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))
# changed the metrics to make a top 5 accuracy predictions
# original metrics was: metrics=['accuracy'], new one is [tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TopKCategoricalAccuracy
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()


# In[19]:


history = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, y_val))


# # Hyper parameter tuning

# In[20]:


import tensorflow as tf
import keras_tuner as kt


# In[21]:


def model_builder(hp):
    '''
    Args:
    hp - Keras tuner object
    '''
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(Dropout(0.3))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(num_classes, activation='softmax'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    return model


# In[22]:


# Instantiate the tuner
tuner = kt.Hyperband(model_builder, # the hypermodel
                     objective='accuracy', # objective to optimize
max_epochs=5,
factor=3, # default
directory='dir', # directory to save logs 
project_name='khyperband_ML')


# In[23]:


tuner.search_space_summary() 


# In[24]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
# Perform hypertuning
tuner.search(X_train, y_train, epochs=5, batch_size = 35,  validation_data=(X_val, y_val),  callbacks=[stop_early])


# In[25]:


best_hp=tuner.get_best_hyperparameters()[0]
best_hp


# In[26]:


# Build the model with the optimal hyperparameters
h_model = tuner.hypermodel.build(best_hp)
h_model.summary()
h_model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))


# # Results and Confusion matrix
# https://stackoverflow.com/questions/65618137/confusion-matrix-for-multiple-classes-in-python: link cm

# In[27]:


from sklearn.metrics import mean_absolute_error, r2_score
# results base model
score = model.evaluate(test_cnn, test_y_labels, verbose=1)
result = model.predict(test_cnn)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("mean absolute error:", mean_absolute_error(test_y_labels, result))


# In[28]:


# results tuned model
score_tuned = h_model.evaluate(test_cnn, test_y_labels, verbose=1)
result_tuned = h_model.predict(test_cnn)
print("Test loss:", score_tuned[0])
print("Test accuracy:", score_tuned[1])
print("mean absolute error:", mean_absolute_error(test_y_labels, result_tuned))


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

y_pred=np.argmax(result, axis=1)
y_test=np.argmax(test_y_labels, axis=1)
cm = confusion_matrix(y_test, y_pred)

## Get Class Labels
le = preprocessing.LabelEncoder()
le.fit_transform(y_test)
labels = le.classes_
class_names = labels

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(class_names, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Refined Confusion Matrix', fontsize=20)


# In[34]:


from sklearn.metrics import precision_score
precisionScore_sklearn_macroavg = precision_score(y_test, y_pred, average='macro')
precisionScore_sklearn_macroavg


# In[32]:


score_cm = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score_cm)
print()
#accuracy per class
acc_per_class = cm.max(axis=1)/cm.sum(axis=1)
for i in range(len(acc_per_class)):
    print("for class", class_names[i], "the accuracy is:", acc_per_class[i])


# # Task 2

# In[35]:


# loading the data
test_dataT2 = np.load('test_images_task2.npy')
print("Shape of test data: ", test_dataT2.shape)


# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


get_ipython().run_line_magic('matplotlib', 'inline')
from skimage import data, io, color, util, morphology, exposure, feature,filters, img_as_float
from skimage.io import imread, imshow
from skimage.color import rgb2gray, label2rgb
from skimage.util import img_as_ubyte
from skimage.exposure import histogram 
from skimage.morphology import dilation, erosion, disk, opening, closing, area_closing, area_opening, reconstruction, extrema, binary_closing,remove_small_objects, white_tophat, black_tophat,binary_dilation
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks, probabilistic_hough_line
from skimage.draw import circle_perimeter
from skimage.filters import rank, threshold_otsu, roberts, sobel, sobel_h, sobel_v, prewitt, prewitt_v, prewitt_h
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import square, rectangle, diamond, disk, cube,  octahedron, ball, star, octagon 
from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing
from scipy.ndimage.morphology import binary_fill_holes

import scipy.ndimage.filters
from scipy import ndimage as ndi
from mpl_toolkits.mplot3d import Axes3D  


from skimage.morphology import disk 
from skimage.filters import median 
from skimage.util import random_noise, img_as_float


seed = 22


# # Function for image processing

# In[44]:


def ImagePreprocessing(file):
    '''Apply thresholding'''
    im = file.astype(float)                
    im = im-im.min()                    
    im = im/im.max()
    block_size = 9
    imbw = im > filters.threshold_local(im, block_size,method = 'mean')
    
    '''Apply median filter to remove noise'''
    denoised_image = median(  imbw  , selem =np.ones((1, 1)))
    
    '''Apply Adaptive Histogram Equalization'''
    im = denoised_image.astype(float)
    im = im-im.min()
    im = im/im.max()
    img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.1)
    
    '''Apply Contrast stretching '''
    im=  img_adapteq
    im = im.astype(float)
    im = im-im.min()
    im = im/im.max()
    L = np.percentile(im,20) 
    H = np.percentile(im, 60)    
    img_rescale = exposure.rescale_intensity(im, in_range=(L, H))
    
    '''Remove small object '''
    im =    img_rescale #img_eq
    im = im.astype(float)
    im = im-im.min()
    im = im/im.max()
    im = im < 0.5 #
    remove_object = remove_small_objects(im,15, connectivity=1)  #fixed
    '''White tophat filtering ''' 
    whitetophat = white_tophat( remove_object, disk(30)) #black_tophat
    
    '''Binary Dilation ''' 
    final = binary_dilation( remove_object, disk(1))
    
    '''Remove Small Objects ''' 
    remove_object = remove_small_objects(final,210, connectivity=1) 
 
    plt.imshow(    remove_object   , cmap=plt.get_cmap('gray'))
    plt.axis('off')
    return    remove_object 
 


# # Functions image cropping

# In[37]:


# Add equal padding of black pixels horizontally (from left side and right side)
def AdjustHorizontalShape(image):
    result = np.zeros((image.shape[0], 28))
    H_offset = int((28 - image.shape[1]) / 2)
    result[: image.shape[0],H_offset : image.shape[1] + H_offset] = image
    return result


# In[38]:


# Add equal padding of black pixels vertically (from top and bottom)
def AdjustVerticalShape(image):
    result = np.zeros((28, image.shape[1]))
    V_offset = int((28 - image.shape[0]) / 2)
    result[V_offset: image.shape[0] + V_offset,: image.shape[1]] = image
    return result


# In[39]:


def Cropping(image):
    # Label image regions
    label_image = label(image)

    cropped_regions = {}
    for region in regionprops(label_image):
        # Take proper regions with large enough areas
        if region.area >= 200:
            # Add region into a dictionary where key is minX coordinate needed for sorting
            minY, minX, maxY, maxX = region.bbox
            cropped_regions[minX] = image[minY: maxY, minX: maxX]
    
    # Sort dictionary by keys
    sorted_regions = []
    for item in sorted(cropped_regions.items()):
        sorted_regions.append(item[1])
        # Resize regions to 28 * 28 pixels to pass them into a recognition model
    resized_regions1 = []
    for sr in sorted_regions:
        if sr.shape[0] > 28:
            resized_regions1.append(resize(sr, (28, sr.shape[1])))
        else:
            resized_regions1.append(sr)
    
    resized_regions2 = []
    for rr1 in resized_regions1:
        if rr1.shape[1] > 28:
            resized_regions2.append(resize(rr1, (rr1.shape[0], 28)))
        else:
            resized_regions2.append(rr1)
    
    
    resized_regions3 = []
    for rr2 in resized_regions2:
        if rr2.shape[1] <28:
            resized_regions3.append(AdjustHorizontalShape(rr2))
        else:
            resized_regions3.append(rr2)
            
    resized_regions4 = []
    for rr3 in resized_regions3:
        if rr3.shape[0] <28:
            resized_regions4.append(AdjustVerticalShape(rr3))
        else:
            resized_regions4.append(rr3)
            
   # Reshape regions to (28, 28, 1) to pass them into the following recognition model
    reshaped_regions = []
    for rr4 in resized_regions4:
        reshaped_regions.append(rr4.reshape(28, 28 , 1).astype('float32'))
    
    # Generate numpy array to apply them for prediction
    cropped_images = np.array(reshaped_regions)

    return cropped_images  


#reference: 
#https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934


# # functions padding and slicing

# In[40]:


# padding leading zero

import numpy as np

def pad(value):
    try:
        return '{0:0>2}'.format(int(value))
    except:
        return value

leadingzero = np.vectorize(pad)


# In[41]:


def slice_per(source, step):
    return [source[i::step] for i in range(step)]

#https://stackoverflow.com/questions/26945277/how-to-split-python-list-every-nth-element


# In[42]:


# Function to convert  
def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # return string  
    return (str1.join(s))


# # Test functions with images

# In[45]:


from skimage.transform import resize



test_image = test_dataT2[1].astype('float32')
image = ImagePreprocessing(test_image)
sample_images =  Cropping(image)
test_images = []
for ci in sample_images:
    test_images.append(ci.reshape(28,28))

fig, axs = plt.subplots(1, len(test_images))

for i in range(0, len(test_images)):
    axs[i].imshow(test_images[i], cmap=plt.get_cmap('gray'))
    axs[i].set_xticks([])
    axs[i].set_yticks([])
plt.show()


plt.imshow(test_image, cmap=plt.get_cmap('gray'))
plt.show()


# In[46]:


test_image = test_dataT2[209].astype('float32')
plt.imshow(test_image, cmap=plt.get_cmap('gray'))
plt.show()
ImagePreprocessing(test_image)


# In[47]:


test_image = test_dataT2[9585].astype('float32')
plt.imshow(test_image, cmap=plt.get_cmap('gray'))
plt.show()
ImagePreprocessing(test_image)


# In[48]:


test_image = test_dataT2[4743].astype('float32')
plt.imshow(test_image, cmap=plt.get_cmap('gray'))
plt.show()
ImagePreprocessing(test_image)


# In[49]:




from datetime import datetime
    
test_dataT2_new = test_dataT2 
counter = 0

Prediction = []


print('{}'.format(datetime.now()))
for image in test_dataT2_new:

  
    preprocessedimage = ImagePreprocessing(image) 
    #Cropping the regions of the image
    sample_images = Cropping(preprocessedimage)
    #for i in range(0, 5):
    
    
        #probas = model.predict_proba(sample_images)
    predictions = model.predict(sample_images)

    # predict the top n labels on validation dataset
    #classifier.probability = True
    
   
    top5_accuracy = []
    #for k in range(0, 5):

    #Identify the indexes of the top predictions
    top_n_predictions = np.argsort(predictions , axis = 1)[:,-5:]
    np_predictions=np.array(top_n_predictions)
    trans_prediction = np_predictions.transpose()

        
    for num in  trans_prediction:    
                top_n_predictions = leadingzero(num) #padding leading zero
                stringtop5_accuracy=listToString(top_n_predictions)
                
                top5_accuracy.append(stringtop5_accuracy)
                
    Prediction.append(top5_accuracy)        
    


    
    
    
print('{}'.format(datetime.now()))


# In[50]:


df=pd.DataFrame(Prediction)
df


# In[51]:


df.to_csv("prediction_final.csv", index=False)

