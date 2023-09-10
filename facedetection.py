import cv2
import numpy as np
import os
import csv

# Create or open a CSV file for writing
csv_file = open('recognized_faces.csv', mode='w', newline='')

# Create a CSV writer object
csv_writer = csv.writer(csv_file)

# Write the header row (if the file is new)
csv_writer.writerow(['Face ID', 'Name'])

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index], output[1][index]

cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = []
labels = []

class_id = 0  # Labels for the given file
names = {}  # Mapping btw id - name

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Create a mapping btw class_id and name
        names[class_id] = fx[:-4]
        print("Loaded " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Create Labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

# Confidence Threshold for Recognition
confidence_threshold = 5  # Adjust this threshold as needed

# Testing
while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if (len(faces) == 0):
        continue

    for face in faces:
        x, y, w, h = face

        # Get the face ROI
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Predicted Label (out)
        out, confidence = knn(trainset, face_section.flatten())

        # Display "Unknown" if confidence is below the threshold
        if confidence < confidence_threshold:
            pred_name = "Unknown"
        else:
            pred_name = names[int(out)]

        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Write the recognized face ID and name to the CSV file
        csv_writer.writerow([int(out), pred_name])

    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

csv_file.close()

cap.release()
cv2.destroyAllWindows()


