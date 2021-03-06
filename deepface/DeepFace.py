import math
import warnings

from keras.preprocessing import image

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import cv2
from PIL import Image
from deepface import Age, Race


def distance(a, b):
    x1 = a[0];
    y1 = a[1]
    x2 = b[0];
    y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def get_opencv_path():
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    face_detector_path = path + "/data/haarcascade_frontalface_default.xml"
    eye_detector_path = path + "/data/haarcascade_eye.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ", face_detector_path,
                         " violated.")

    return path + "/data/"


def detectFace(img, target_size=(224, 224), grayscale=False, enforce_detection=True):
    img_path = ""

    # -----------------------

    exact_image = False
    if type(img).__module__ == np.__name__:
        exact_image = True

    # -----------------------

    opencv_path = get_opencv_path()
    face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
    eye_detector_path = opencv_path + "haarcascade_eye.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ", face_detector_path,
                         " violated.")

    # --------------------------------

    face_detector = cv2.CascadeClassifier(face_detector_path)
    eye_detector = cv2.CascadeClassifier(eye_detector_path)

    if exact_image != True:  # image path passed as input

        if os.path.isfile(img) != True:
            raise ValueError("Confirm that ", img, " exists")

        img = cv2.imread(img)

    img_raw = img.copy()

    # --------------------------------

    faces = []

    try:
        faces = face_detector.detectMultiScale(img, 1.3, 5)
    except:
        pass

    # print("found faces in ",image_path," is ",len(faces))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]
        detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        # ---------------------------
        # face alignment

        eyes = eye_detector.detectMultiScale(detected_face_gray)

        if len(eyes) >= 2:
            # find the largest 2 eye
            base_eyes = eyes[:, 2]

            items = []
            for i in range(0, len(base_eyes)):
                item = (base_eyes[i], i)
                items.append(item)

            df = pd.DataFrame(items, columns=["length", "idx"]).sort_values(by=['length'], ascending=False)

            eyes = eyes[df.idx.values[0:2]]

            # -----------------------
            # decide left and right eye

            eye_1 = eyes[0];
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            # -----------------------
            # find center of eyes

            left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
            left_eye_x = left_eye_center[0];
            left_eye_y = left_eye_center[1]

            right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
            right_eye_x = right_eye_center[0];
            right_eye_y = right_eye_center[1]

            # -----------------------
            # find rotation direction

            if left_eye_y > right_eye_y:
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1  # rotate same direction to clock
            else:
                point_3rd = (left_eye_x, right_eye_y)
                direction = 1  # rotate inverse direction of clock

            # -----------------------
            # find length of triangle edges

            a = distance(left_eye_center, point_3rd)
            b = distance(right_eye_center, point_3rd)
            c = distance(right_eye_center, left_eye_center)

            # -----------------------
            # apply cosine rule

            if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

                cos_a = (b * b + c * c - a * a) / (2 * b * c)
                angle = np.arccos(cos_a)  # angle in radian
                angle = (angle * 180) / math.pi  # radian to degree

                # -----------------------
                # rotate base image

                if direction == -1:
                    angle = 90 - angle

                img = Image.fromarray(img_raw)
                img = np.array(img.rotate(direction * angle))

                # you recover the base image and face detection disappeared. apply again.
                faces = face_detector.detectMultiScale(img, 1.3, 5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    detected_face = img[int(y):int(y + h), int(x):int(x + w)]

        # -----------------------

        # face alignment block end
        # ---------------------------

        # face alignment block needs colorful images. that's why, converting to gray scale logic moved to here.
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        detected_face = cv2.resize(detected_face, target_size)

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # normalize input in [0, 1]
        img_pixels /= 255

        return img_pixels

    else:

        if (exact_image == True) or (enforce_detection != True):

            if grayscale == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, target_size)
            img_pixels = image.img_to_array(img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            return img_pixels
        else:
            raise ValueError(
                "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")


def analyze(img_path, actions=[], models={}, enforce_detection=True):
    if type(img_path) == list:
        img_paths = img_path.copy()
        bulkProcess = True
    else:
        img_paths = [img_path]
        bulkProcess = False

    # ---------------------------------

    # if a specific target is not passed, then find them all
    if len(actions) == 0:
        actions = ['age', 'race']

    print("Actions to do: ", actions)

    # ---------------------------------
    if 'age' in actions:
        if 'age' in models:
            print("already built age model is passed")
            age_model = models['age']
        else:
            age_model = Age.loadModel()

    if 'race' in actions:
        if 'race' in models:
            print("already built race model is passed")
            race_model = models['race']
        else:
            race_model = Race.loadModel()
    # ---------------------------------

    resp_objects = []

    global_pbar = tqdm(range(0, len(img_paths)), desc='Analyzing')

    # for img_path in img_paths:
    for j in global_pbar:
        img_path = img_paths[j]

        resp_obj = "{"

        # TO-DO: do this in parallel

        pbar = tqdm(range(0, len(actions)), desc='Finding actions')

        action_idx = 0
        img_224 = None  # Set to prevent re-detection
        # for action in actions:
        for index in pbar:
            action = actions[index]
            pbar.set_description("Action: %s" % (action))

            if action_idx > 0:
                resp_obj += ", "

            if action == 'age':
                if img_224 is None:
                    img_224 = detectFace(img_path, target_size=(224, 224), grayscale=False,
                                         enforce_detection=enforce_detection)  # just emotion model expects grayscale images
                # print("age prediction")
                age_predictions = age_model.predict(img_224)[0, :]
                apparent_age = Age.findApparentAge(age_predictions)

                resp_obj += "\"age\": %s" % (apparent_age)

            elif action == 'race':
                if img_224 is None:
                    img_224 = detectFace(img_path, target_size=(224, 224), grayscale=False,
                                         enforce_detection=enforce_detection)  # just emotion model expects grayscale images
                race_predictions = race_model.predict(img_224)[0, :]
                race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

                sum_of_predictions = race_predictions.sum()

                race_obj = "\"race\": {"
                for i in range(0, len(race_labels)):
                    race_label = race_labels[i]
                    race_prediction = 100 * race_predictions[i] / sum_of_predictions

                    if i > 0: race_obj += ", "

                    race_obj += "\"%s\": %s" % (race_label, race_prediction)

                race_obj += "}"
                race_obj += ", \"dominant_race\": \"%s\"" % (race_labels[np.argmax(race_predictions)])

                resp_obj += race_obj

            action_idx = action_idx + 1

        resp_obj += "}"

        resp_obj = json.loads(resp_obj)

        if bulkProcess == True:
            resp_objects.append(resp_obj)
        else:
            return resp_obj

    if bulkProcess == True:
        resp_obj = "{"

        for i in range(0, len(resp_objects)):
            resp_item = json.dumps(resp_objects[i])

            if i > 0:
                resp_obj += ", "

            resp_obj += "\"instance_" + str(i + 1) + "\": " + resp_item
        resp_obj += "}"
        resp_obj = json.loads(resp_obj)
        return resp_obj
