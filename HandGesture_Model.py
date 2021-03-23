
import tensorflow as tf
import os
import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import Sequential


DATADIR = r"G:/Projects/HGR/leapGestRecog"

temp1 = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
temp2 = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']


img_size = 128
X = []
y = []


i = 0
for t1 in temp1[:1]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i += 1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))

plt.imshow(X[0])

plt.imshow(X[1999])

print(y[290:650])

i = 0
for t1 in temp1[1:2]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i+=1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))


i = 0
for t1 in temp1[2:3]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i += 1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))


i = 0
for t1 in temp1[3:4]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i+=1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))


i = 0
for t1 in temp1[4:5]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i+=1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))

i = 0
for t1 in temp1[5:6]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i+=1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))

i = 0
for t1 in temp1[6:7]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i+=1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))


i = 0
for t1 in temp1[7:8]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i+=1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))


i = 0
for t1 in temp1[8:9]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i+=1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))

i = 0
for t1 in temp1[9:10]:
    DATADIR_final = os.path.join(DATADIR,t1)
    for t2 in temp2:
        path = os.path.join(DATADIR_final, t2) 
        print(path)
        i+=1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_image = cv2.resize(img_array, (img_size, img_size))
            X.append(new_image)
            y.append(i-1)
        
print(len(X))


X = np.array(X)

y = np.array(y)

y = y.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


len(X_train)
len(y_train)


classification = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved','c','down']


plt.imshow(X[0])
print(X_train[0])
print(y_train)

y_train_tc = to_categorical(y_train)
y_test_tc= to_categorical(y_test)

X_train = X_train / 255
X_test = X_test / 255


# Create the models architecture
model = Sequential()

# Add the first layer
model.add(Conv2D(128, (4, 4),strides = (2, 2), activation = 'relu', input_shape = (128, 128, 3)))

# Add a pooling Layer
model.add(MaxPooling2D(pool_size = (2, 2)))

# Add the second layer
model.add(Conv2D(64, (4, 4), activation = 'relu'))

# Add a pooling Layer
model.add(MaxPooling2D(pool_size = (2, 2)))

# Add a flattening layer
model.add(Flatten())

# Add a layer with 256 neurons
model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

# Add a layer with 128 neurons
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

# Add a layer with 64 neurons
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

# Add a layer with 10 neurons
model.add(Dense(10, activation='softmax'))

# print(model)

# Model Compilation
model.compile(loss = 'categorical_crossentropy',optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),metrics = ['accuracy'])
hist = model.fit(X_train,y_train_tc, epochs=5, validation_data=(X_test, y_test_tc), verbose=1)


m = model.predict(X_train[:20])
for i in range(20):
    print(y_train[i], np.argmax(m[i], axis = 0))

predictions = model.predict(X_test)


# print(np.argmax(predictions, axis = 1))
# print(y_test)


model.evaluate(X_test, y_test_tc, batch_size = 128)
model.save('saved_model/my_model')


