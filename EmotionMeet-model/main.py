
import cv2,dlib
import sys
from renderFace import renderFace, renderFace2, renderFace3, renderFace4
import json
import numpy as np
from xception import MiniXception

PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"
RESIZE_HEIGHT = 240 #480
SKIP_FRAMES = 2
DATA_PATH = "./data"
keys = list(range(0, 69))


try:

    # Create an imshow window
    winName = "Fast Facial Landmark Detector"

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    '''
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    capwriter = cv2.VideoWriter('output.avi',fourcc, fps, (w,h))
    '''

    # Check if OpenCV is able to read feed from camera
    if (cap.isOpened() is False):
        print("Unable to connect to camera")
        sys.exit()

    # Just a place holder. Actual value calculated after 100 frames.
    fps = 30.0

    # Get first frame
    ret, im = cap.read()

    # We will use a fixed height image as input to face detector
    if ret == True:
        height = im.shape[0]
        # calculate resize scale
        RESIZE_SCALE = float(height)/RESIZE_HEIGHT
        size = im.shape[0:2]
    else:
        print("Unable to read frame")
        sys.exit()

    # Load face detection and pose estimation models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    classifier = MiniXception((48, 48, 1), 7, weights='FER')
    #class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    #class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
    # initiate the tickCounter
    t = cv2.getTickCount()
    count = 0

    # Grab and process frames until the main window is closed by the user.
    while(True):
        if count==0:
            t = cv2.getTickCount()
        # Grab a frame
        ret, im = cap.read()
        imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # create imSmall by resizing image by resize scale
        imSmall= cv2.resize(im, None, fx = 1.0/RESIZE_SCALE, fy = 1.0/RESIZE_SCALE, interpolation = cv2.INTER_LINEAR)
        imSmallDlib = cv2.cvtColor(imSmall, cv2.COLOR_BGR2RGB)
        # Stop the program.
        # Process frames at an interval of SKIP_FRAMES.
        # This value should be set depending on your system hardware
        # and camera fps.
        # To reduce computations, this value should be increased
        if (count % SKIP_FRAMES == 0):
            # Detect faces
            faces = detector(imSmallDlib,0)

        # Iterate over faces
        for face in faces:
            # Since we ran face detection on a resized image,
            # we will scale up coordinates of face rectangle
            newRect = dlib.rectangle(int(face.left() * RESIZE_SCALE),
                                   int(face.top() * RESIZE_SCALE),
                                   int(face.right() * RESIZE_SCALE),
                                   int(face.bottom() * RESIZE_SCALE))

            faceRet = [(int(face.left() * RESIZE_SCALE), int(face.top() * RESIZE_SCALE)),
                       (int(face.right() * RESIZE_SCALE), int(face.top() * RESIZE_SCALE)),
                       (int(face.left() * RESIZE_SCALE), int(face.bottom() * RESIZE_SCALE)),
                       (int(face.right() * RESIZE_SCALE), int(face.bottom() * RESIZE_SCALE))]
            left = int(face.left() * RESIZE_SCALE)
            right = int(face.right() * RESIZE_SCALE)
            top = int(face.top() * RESIZE_SCALE)
            bottom = int(face.bottom() * RESIZE_SCALE)

            imFace = im[top:bottom,left:right]
            try:
                imFace = cv2.cvtColor(imFace, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('./data/test3.jpg', imFace)
                imFace = cv2.resize(imFace, dsize=(48, 48), interpolation = cv2.INTER_LINEAR)
                imFace = np.reshape(imFace,(1,48,48,1))
                face_class = classifier(imFace)
                face_class = class_names[np.argmax(face_class)]
            except:
                continue
            cv2.imwrite('./data/test.jpg', im)
            # Find face landmarks by providing reactangle for each face
            landmarks = predictor(imDlib, newRect)
            landmark_list = []

            # append (x, y) in landmark_list
            for p in landmarks.parts():
                landmark_list.append([p.x, p.y])

            with open("./data/test.json", "w") as json_file:
            	key_val = [keys, landmark_list + [face_class]]
            	landmark_dict = dict(zip(*key_val))
            	json_file.write(json.dumps(landmark_dict))
            	json_file.write('\n')

            # Draw facial landmarks
            renderFace(im, landmarks)
            renderFace2(im, landmarks)
            renderFace3(im, faceRet)
            renderFace4(im, face_class,faceRet[1])
            cv2.imwrite('./data/test2.jpg', im)


        # Put fps at which we are processinf camera feed on frame
        cv2.putText(im, "{0:.2f}-fps".format(fps), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
        # Display it all on the screen
        cv2.imshow(winName, im)
        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF

        if key==27:  # ESC
            # If ESC is pressed, exit.
            #sys.exit()
            break

        # increment frame counter
        count = count + 1
        # calculate fps at an interval of 100 frames
        if (count == 100):
            t = (cv2.getTickCount() - t)/cv2.getTickFrequency()
            fps = 100.0/t
            count = 0
    cv2.destroyAllWindows()
    cap.release()
    sys.exit()

except Exception as e:
    print(e)
