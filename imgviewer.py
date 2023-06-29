import PySimpleGUI as sg
import os.path
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
sg.theme('purple')

# Load the pre-trained gender model
gender_model = load_model('./Model/gender_model_pretrained.h5')
gender_ranges = ['male', 'female']

file_list_column = [
    [
        sg.Text("Open Image Folder:")
    ],
    [
        sg.In(size=(20, 1), enable_events=True, key="ImgFolder"),
        sg.FolderBrowse()
    ],
    [
        sg.Text("Choose an image from list:")
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(18, 10), key="ImgList"
        )
    ]
]

image_viewer_column = [
    [sg.Text("Image Input:")],
    [sg.Text(size=(40, 1), key="FilepathImgInput")],
    [sg.Image(key="ImgInputViewer")],
    [sg.Button("Process Image", key="ProcessImage")]
]

image_viewer_column2 = [
    [sg.Text("Processed Image:")],
    [sg.Image(key="ImgOutputViewer")],
    [sg.Text("Gender Prediction:")],
    [sg.Multiline(size=(40, 10), key="GenderPrediction")],

]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeparator(),
        sg.Column(image_viewer_column),
        sg.VSeparator(),
        sg.Column(image_viewer_column2),
    ]
]

window = sg.Window("Mini Image Editor", layout)

while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WINDOW_CLOSED:
        break

    if event == "ImgFolder":
        folder = values["ImgFolder"]

        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".jpg", ".jpeg"))
        ]
        window["ImgList"].update(fnames)

    elif event == "ImgList":
        try:
            filename = os.path.join(values["ImgFolder"], values["ImgList"][0])
            window["FilepathImgInput"].update(filename)
            window["ImgInputViewer"].update(filename=filename)
        except:
            pass

    elif event == "ProcessImage":
        try:
            filename = os.path.join(values["ImgFolder"], values["ImgList"][0])

            # Process the selected image
            test_image = cv2.imread(filename)
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            gender_predictions = []

            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(test_image, (x, y),
                              (x+w, y+h), (203, 12, 255), 2)

                img_gray = gray[y:y+h, x:x+w]

                gender_img = cv2.resize(
                    img_gray, (100, 100), interpolation=cv2.INTER_AREA)
                gender_image_array = np.array(gender_img)
                gender_input = np.expand_dims(gender_image_array, axis=0)
                output_gender = gender_ranges[np.argmax(
                    gender_model.predict(gender_input))]

                gender_predictions.append(output_gender)

                col = (0, 255, 0)
                # Adjust the font scale based on face size
                font_scale = max(w, h) / 30
                cv2.putText(test_image, str(i+1), (x, y+h+30),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 2)

            window["ImgOutputViewer"].update(
                data=cv2.imencode('.png', test_image)[1].tobytes())

            gender_output_str = '\n'.join(
                [f"{i+1}: {gender}" for i, gender in enumerate(gender_predictions)])
            window["GenderPrediction"].update(gender_output_str)

        except:
            pass

window.close()
