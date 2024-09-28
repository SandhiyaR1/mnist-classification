# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.The MNIST dataset is a collection of handwritten digits.The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
## Neural Network Model
![image](https://github.com/user-attachments/assets/757eaea4-1d7d-4ea0-94d3-e521faf38440)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries,Download and load the dataset.
### STEP 2:
 Scale the dataset between it's min and max values, Using one hot encode, encode the categorical values.
### STEP 3:
 Split the data into train and test, Build the convolutional neural network model.
 ### STEP 4:
 Train the model with the training data, Plot the performance plot.
 ### STEP 5:
 Evaluate the model with the testing data, Fit the model and predict the single input.
## PROGRAM

### Name:SANDHIYA R
### Register Number:212222230129
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
x_test.shape
single_image = x_train[500]
single_image
print(y_train[500])
plt.imshow(single_image,cmap='gray')
y_train.shape
x_train.min()
x_train.max()
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
x_train_scaled.min()
x_train_scaled.max()
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = x_train[500]
plt.imshow(single_image,cmap='gray')
x_train_scaled = x_train_scaled.reshape(-1,28,28,1)
x_test_scaled = x_test_scaled.reshape(-1,28,28,1)
model =keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(x_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
print(" -SANDHIYA R")
metrics[['loss','val_loss']].plot()
print(" -SANDHIYA R")
x_test_predictions = np.argmax(model.predict(x_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = x_train[550]
type(img)
plt.imshow(img,cmap='gray')
img = image.load_img('image.png')
tensor_img = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(tensor_img,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/654db13d-8ea7-4a47-b64f-808a2125b55e)

![image](https://github.com/user-attachments/assets/0f4a3349-4368-4d5c-8b12-d4795bbbe7f3)

### Classification Report
![image](https://github.com/user-attachments/assets/aaeda8af-2304-4298-bebf-7063122681c7)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/db2819aa-dd0e-4f92-8395-bcfe26de181b)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/3e43a0f6-70dd-4fd5-8fe1-5b6b62d1a76c)

![image](https://github.com/user-attachments/assets/29bcfb81-02d9-4587-9930-ac18a9d99dbc)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
