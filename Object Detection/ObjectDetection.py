from keras.models import load_model
import numpy as np

import cv2

model = load_model('keras_model.h5')

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#Specifying the path of the image.
image = cv2.imread('ImagesTrain/appleTrain.jpg')
img = cv2.resize(image, (224, 224))


image_array = np.asarray(img)

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_array

prediction = model.predict(data)

for i in prediction:
    if i[0] > 0.7:
        text = classLabels[0]
    if i[1] > 0.7:
        text = classLabels[1]
    if i[2] > 0.7:
        text = classLabels[2]
    if i[3] > 0.7:
        text = classLabels[3]
    if i[4] > 0.7:
        text = classLabels[4]
    if i[5] > 0.7:
        text = classLabels[5]
    img = cv2.resize(img, (500, 500))
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(img, (50,50), (400,400), (255,0,0), thickness=2)
cv2.imshow('img', img)
cv2.waitKey(0)
