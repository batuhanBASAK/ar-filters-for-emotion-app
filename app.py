import numpy as np
import cv2
import mediapipe as mp
import pickle

cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print('Camera counldn\'t been opened')
    exit(1)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_selfie = mp.solutions.selfie_segmentation
model =  mp_selfie.SelfieSegmentation(model_selection=0)

labels = ['angry', 'happy', 'sad']

with open('model.pickle', 'rb') as handle:
    clf = pickle.load(handle)

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
            X = []
            aux = []
            for l in facial_landmarks.landmark:
                # drawing each landmark on frame
                # pos_x = int(l.x * frame_width)
                # pos_y = int(l.y * frame_height)
                # cv2.circle(frame, (pos_x, pos_y), 2, (0, 255, 0), -1)

                aux.append(l.x)
                aux.append(l.y)
                aux.append(l.z)
            X.append(aux)
            X = np.asarray(X)
            y_pred = np.argmax(clf.predict(X.reshape(1, -1)))
            print(labels[y_pred])

            # segmentate the person from background
            segmentation_results = model.process(rgb_frame)
            # change background respect to emotion of person
            if labels[y_pred] == 'sad':
                background = cv2.imread('./background-images/sad.jpg')
            elif labels[y_pred] == 'happy':
                background = cv2.imread('./background-images/happy.jpg')
            else: # angry
                background = cv2.imread('./background-images/angry.jpg')
            background = cv2.resize(background, (frame_width, frame_height))
            mask = np.stack((segmentation_results.segmentation_mask,)*3, axis=-1) > 0.4 
            frame = np.where(mask, frame, background)

            # Display the predicted emotion on top left corner of frame
            cv2.rectangle(frame, (0, 0), (150, 50), (0, 0, 0), -1)
            text = labels[y_pred]
            cv2.putText(frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)



    
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break


    # Display the frame
    cv2.imshow('AR Emotion Filter Program', frame)



    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()