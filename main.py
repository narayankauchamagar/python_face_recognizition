import cv2
import os
import numpy as np
from PIL import Image
import json

_DATA_PATH = "./data"
_TRAIN_IMAGE_COUNT = 300
_FACE_CASCADE_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_CLASSIFIER = cv2.face.LBPHFaceRecognizer_create()
_USER_MAP = {}
_USER_RECORD_FILE = "user_record.txt"


def face_cropped(image_frame, face_classifier):
    # convert image to gray scale
    gray_image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    # face detection on the grayscale image
    # scaling factor = 1.3
    # minimum neighbor = 5
    faces = face_classifier.detectMultiScale(gray_image, 1.3, 5)
    if faces is ():
        return None
    cropped_face = []
    for (x, y, w, h) in faces:
        cropped_face = image_frame[y:y + h, x:x + w]
    return cropped_face


def write_user_record_to_file(user_records: map):
    with open(_USER_RECORD_FILE, 'w') as convert_file:
        convert_file.write(json.dumps(user_records))


def read_user_record_from_file():
    if os.path.exists(_USER_RECORD_FILE) and os.path.isfile(_USER_RECORD_FILE):
        # reading the data from the file
        with open(_USER_RECORD_FILE) as f:
            data = f.read()

        print("Data type before reconstruction : ", type(data))
        # reconstructing the data as a dictionary
        return json.loads(data)
    else:
        return {}


def generate_user_data_and_train_data():
    _USER_MAP = read_user_record_from_file()
    print("Start: collect user count")
    count: int = int(input("Enter User count for data collection:"))
    x = 0
    while x < count:
        user_name = input("User's Name: ")
        user_id = int(input("Enter user id"))
        if _USER_MAP.keys().__contains__(user_id):
            print("User id already taken: over-riding data")
        _USER_MAP[user_id] = user_name
        generate_dataset(user_id)
        x += 1
    write_user_record_to_file(_USER_MAP)
    if count > 0:
        # train images of users
        train_classifier("./data")
    return _USER_MAP


def generate_dataset(user_id):
    if not (os.path.exists(_DATA_PATH) & os.path.isdir(_DATA_PATH)):
        os.makedirs(_DATA_PATH)
    camera = cv2.VideoCapture(1)
    img_id = 0

    while True:
        returned, frame = camera.read()
        face = face_cropped(frame, _FACE_CASCADE_CLASSIFIER)
        if face is not None:
            img_id += 1
            face = cv2.resize(face, (200, 200))
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "./data/user." + str(user_id) + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, gray_face)

            cv2.imshow("Cropped face", face)
            if cv2.waitKey(1) == 13 or int(img_id) == _TRAIN_IMAGE_COUNT:
                break

    camera.release()
    cv2.destroyAllWindows()
    # issue for un
    for i in range(1, 5):
        cv2.waitKey(1)


def train_classifier(data_dir):
    # all image paths
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    # print(image_paths)
    faces = []
    ids = []

    for image in image_paths:
        if os.path.isfile(image) & image.endswith(".jpg"):
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')

            # extract image id from image path
            id = os.path.split(image)[1].split(".")[1]
            faces.append(imageNp)
            ids.append(int(id))

    ids = np.array(ids)

    # Train the classifier and save
    # or use EigenFaceRecognizer by replacing above line with
    # face_recognizer = cv2.face.createEigenFaceRecognizer()
    # or use FisherFaceRecognizer by replacing above line with
    # face_recognizer = cv2.face.createFisherFaceRecognizer()
    _CLASSIFIER.train(faces, ids)
    _CLASSIFIER.write("classifier.xml")


def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE_CLASSIFIER.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, [], []

    detected_face = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detected_face = img[y:y + h, x:x + w]
        detected_face = cv2.resize(detected_face, (200, 200))
    return img, detected_face, faces


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # capture image from camera and generate images
    _USER_MAP = generate_user_data_and_train_data()

    # detect and recgonize image
    classifier = cv2.face.LBPHFaceRecognizer_create()
    classifier.read("classifier.xml")
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        image, face, features = face_detector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            user_id, result = classifier.predict(face)
            confidence = 0
            if result < 500:
                confidence = int(100 * (1 - result / 300))
            if confidence > 77:
                x = features[0, 0]
                y = features[0, 1]
                cv2.putText(image, _USER_MAP.get(str(user_id)), (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass
        if cv2.waitKey(1) == 13 or cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    # issue for un
    for i in range(1, 5):
        cv2.waitKey(1)
