import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

labels = np.load("./labels.npy")
dataset = np.load("./np_dataset.npy")
names = np.load("./names.npy")
labels_int = np.load("./labels_intt.npy")


X_train, X_test, y_train, y_test = train_test_split(dataset, labels_int, test_size=0.2)

y_train_ohe = to_categorical(y_train, num_classes=3832)
y_test_ohe = to_categorical(y_test, num_classes=3832)

clear_session()

batch_size = 128
num_classes = 10
epochs = 25
img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('{} train samples, {} test samples'.format(len(X_train), len(X_test)))

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape = input_shape))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3832, activation='softmax'))

model.compile(loss= 'categorical_crossentropy',
              optimizer= 'adam',
              metrics='accuracy')

model.fit(X_train, y_train_ohe,
          batch_size= batch_size,
          epochs= epochs,
          verbose=1,
          validation_data=(X_test, y_test_ohe))

train_score = model.evaluate(X_train, y_train_ohe, verbose=1)
test_score = model.evaluate(X_test, y_test_ohe, verbose=1)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

model.save("./sequential_model.keras")
