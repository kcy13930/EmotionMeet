# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

import cv2,dlib
import sys
from renderFace import renderFace, renderFace2, renderFace3, renderFace4
import json
import numpy as np
from xception import MiniXception
import flask
import io

PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"
RESIZE_HEIGHT = 240 #480
SKIP_FRAMES = 2
DATA_PATH = "./data"
keys = list(range(0, 69))

# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
classifier = MiniXception((48, 48, 1), 7, weights='FER')
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

    #class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    #class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']


def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			im = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification


			# classify the input image and then initialize the list
			# of predictions to return to the client
			height = im.shape[0]
			RESIZE_SCALE = float(height)/RESIZE_HEIGHT
			size = im.shape[0:2]

			imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	        # create imSmall by resizing image by resize scale
			imSmall= cv2.resize(im, None, fx = 1.0/RESIZE_SCALE, fy = 1.0/RESIZE_SCALE, interpolation = cv2.INTER_LINEAR)
			imSmallDlib = cv2.cvtColor(imSmall, cv2.COLOR_BGR2RGB)
			faces = detector(image,0)
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
					preds = np.argmax(classifier(imFace))
					face_class = class_names[preds]
				except:
					continue
				cv2.imwrite('./data/test.jpg', im)

				landmarks = predictor(imDlib, newRect)
				landmark_list = []

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

				cv2.putText(im, "{0:.2f}-fps".format(fps), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
		        # Display it all on the screen
				cv2.imshow(winName, im)
		        # Wait for keypress
				key = cv2.waitKey(1) & 0xFF


			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			r = {"label": face_class, "probability": preds}
			data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run()
