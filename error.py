from IPython.display import Image
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# the following are to do with this interactive notebook code


# this lets you draw inline pictures in the notebooks
from matplotlib import pyplot as plt
import pylab  # this allows you to control figure size
# this controls figure size in the notebook
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# load and evaluate a saved model
export_dir = './Model/gender_model_pretrained.h5'
gender_model = load_model(export_dir)

# summarize model.
gender_model.summary()

gender_ranges = ['male', 'female']

# foto tester
img_path = "./images/wisma.png"

# menampilkan image
pil_img = Image.open(img_path)
plt.imshow(pil_img)
plt.axis('off')
plt.show()


# run test image
test_image = cv2.imread(img_path)
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

i = 0

for (x, y, w, h) in faces:
    i = i+1
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (203, 12, 255), 2)

    img_gray = gray[y:y+h, x:x+w]

#   emotion_img = cv2.resize(img_gray, (48, 48), interpolation = cv2.INTER_AREA)
#   emotion_image_array = np.array(emotion_img)
#   emotion_input = np.expand_dims(emotion_image_array, axis=0)
#   output_emotion= emotion_ranges[np.argmax(emotion_model.predict(emotion_input))]

    gender_img = cv2.resize(img_gray, (100, 100), interpolation=cv2.INTER_AREA)
    gender_image_array = np.array(gender_img)
    gender_input = np.expand_dims(gender_image_array, axis=0)
    output_gender = gender_ranges[np.argmax(
        gender_model.predict(gender_input))]

#   age_image=cv2.resize(img_gray, (200, 200), interpolation = cv2.INTER_AREA)
#   age_input = age_image.reshape(-1, 200, 200, 1)
#   output_age = age_ranges[np.argmax(age_model.predict(age_input))]

    output_str = str(i) + ": " + output_gender
    print(output_str)

    col = (0, 255, 0)

    cv2.putText(test_image, str(i), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
