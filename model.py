import csv
import cv2
import numpy as np
import sklearn

from keras import backend as K
K.set_image_dim_ordering('tf')


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            generatedimages = []
            generatedangles = []
            for imgpath,msrment in batch_samples:
                
                center_image_i = cv2.imread(imgpath.strip())
                center_image = cv2.resize(center_image_i,(200,66), interpolation = cv2.INTER_AREA)
                image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)                
                generatedimages.append(image)
                generatedangles.append(msrment)
                generatedimages.append(cv2.flip(image,1))
                generatedangles.append(-msrment)

            X_train = np.array(generatedimages)
            y_train = np.array(generatedangles)
            yield sklearn.utils.shuffle(X_train, y_train)
           
samples=[]
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        steering_center = float(line[3])
            
        
        correction = 0.2 
        #adjusting steering angles
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        
        samples.append((('./data/'+line[0].strip()),steering_center))
        samples.append((('./data/'+line[1].strip()),steering_left))
        samples.append((('./data/'+line[2].strip()),steering_right))
       
from sklearn.model_selection import train_test_split

#split training/validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D

#Preproccess
model = Sequential()

print('@@@@@@@@@@@@@@@@@@@')
# NVIDIA ARCHITECTURE
model.add(Convolution2D(24,5,5, subsample=(2, 2),input_shape=(66,200,3),  border_mode='valid',activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2, 2), border_mode='valid',activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2, 2), border_mode='valid',activation="relu"))
model.add(Convolution2D(64,3,3, border_mode='valid',activation="relu"))
model.add(Convolution2D(64,3,3, border_mode='valid',activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
          
print('*****************')                   
print(len(train_samples))
history_object = model.fit_generator(generator= train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=7, verbose=1)

model.save('model.h5')

