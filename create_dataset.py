import numpy as np
import cv2
import mediapipe as mp
import pickle

cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print('Camera counldn\'t been opened')
    exit(1)

labels = ['happy', 'sad', 'angry']


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


NUM_OF_DATA_PER_CLASS = 500


X = []
y = []
curr_class = 0
i = 0
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results and results.multi_face_landmarks:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1] 
        for facial_landmarks in results.multi_face_landmarks:
            for l in facial_landmarks.landmark:
                pos_x = int(l.x * frame_width)
                pos_y = int(l.y * frame_height)
                cv2.circle(frame, (pos_x, pos_y), 2, (0, 255, 0), -1)

    key = cv2.waitKey(1) & 0xff
    if key == ord(' '):
        if results and results.multi_face_landmarks:
            frame_height = frame.shape[0]
            frame_width = frame.shape[1] 
            aux = []
            for facial_landmarks in results.multi_face_landmarks:
                for l in facial_landmarks.landmark:
                    aux.append(l.x)
                    aux.append(l.y)
                    aux.append(l.z)
            X.append(aux)
            y.append(labels[curr_class])
            i += 1
            if i == NUM_OF_DATA_PER_CLASS:
                curr_class += 1
                i = 0
            if curr_class == len(labels):
                # save dataset to file
                X = np.asarray(X)
                y = np.asarray(y)
                with open('dataset.pickle', 'wb') as handle:
                    pickle.dump({'X': X, 'y': y}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # quit the while loop
                break
    elif key == ord('q'):
        break

    text = 'current class: ' + labels[curr_class] + ', current image number: ' + str(i)

    cv2.rectangle(frame, (0, 0), (800, 50), (0, 0, 0), -1)

    cv2.putText(frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2) 

    # Display the frame
    cv2.imshow('Face Landmark Detection', frame)



    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()