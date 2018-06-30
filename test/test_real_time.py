import cv2
import numpy as np
from time import sleep
from keras.models import load_model

# Model used
train_model =  "ResNet" # (Inception-v3, Inception-ResNet-v2): Inception,  (ResNet-50): ResNet

# Size of the images
if train_model == "Inception":
    img_width, img_height = 139, 139
elif train_model == "ResNet":
    img_width, img_height = 197, 197

emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Reinstantiate the fine-tuned model (Also compiling the model using the saved training configuration (unless the model was never compiled))
model = load_model('./../trained_models/ResNet-50.h5')

# Create a face cascade
cascPath = './../trained_models/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Sets the video source to the default webcam
	# device:	id of the opened video capturing device (i.e. a camera index). If there is a single camera connected, just pass 0
video_capture = cv2.VideoCapture(0)

def preprocess_input(image):
    image = cv2.resize(image, (img_width, img_height))  # Resizing images for the trained model
    ret = np.empty((img_height, img_width, 3)) 
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis = 0)   # (1, XXX, XXX, 3)

    if train_model == "Inception":
        x /= 127.5
        x -= 1.
        return x
    elif train_model == "ResNet":
        x -= 128.8006   # np.mean(train_dataset)
        x /= 64.6497    # np.std(train_dataset)

    return x

def predict(emotion):
    # Generates output predictions for the input samples
        # x:    the input data, as a Numpy array (None, None, None, 3)
    prediction = model.predict(emotion)
    
    return prediction

while True:
    if not video_capture.isOpened():	# If the previous call to VideoCapture constructor or VideoCapture::open succeeded, the method returns true
        print('Unable to load camera.')
        sleep(5)						# Suspend the execution for 5 seconds
    else:
        sleep(0.5)
        ret, frame = video_capture.read()						# Grabs, decodes and returns the next video frame (Capture frame-by-frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	# Conversion of the image to the grayscale
        
        # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
        	# image:		Matrix of the type CV_8U containing an image where objects are detected
        	# scaleFactor:	Parameter specifying how much the image size is reduced at each image scale
        	# minNeighbors:	Parameter specifying how many neighbors each candidate rectangle should have to retain it
        	# minSize:		Minimum possible object size. Objects smaller than that are ignored
        faces = faceCascade.detectMultiScale(
            gray_frame,
            scaleFactor		= 1.1,
            minNeighbors	= 5,
            minSize			= (30, 30))

        prediction = None
        x, y = None, None

        for (x, y, w, h) in faces:
            ROI_gray = gray_frame[y:y+h, x:x+w] # Extraction of the region of interest (face) from the frame

            # Draws a simple, thick, or filled up-right rectangle
                # img:          Image
                # pt1:          Vertex of the rectangle
                # pt2:          Vertex of the rectangle opposite to pt1
                # rec:          Alternative specification of the drawn rectangle
                # color:        Rectangle color or brightness (BGR)
                # thickness:    Thickness of lines that make up the rectangle. Negative values, like CV_FILLED , 
                #               mean that the function has to draw a filled rectangle
                # lineType:     Type of the line
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            emotion = preprocess_input(ROI_gray)
            prediction = predict(emotion)
            print(prediction[0][0])
            top_1_prediction = emotions[np.argmax(prediction)]

            # Draws a text string
                # img:          Image
                # text:         Text string to be drawn
                # org:          Bottom-left corner of the text string in the image
                # font:         CvFont structure initialized using InitFont()
                # fontFace:     Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX, 
                #               FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX, or FONT_HERSHEY_SCRIPT_COMPLEX,
                #               where each of the font ID's can be combined with FONT_ITALIC to get the slanted letters
                # fontScale:    Font scale factor that is multiplied by the font-specific base size
                # color:        Text color
                # thickness:    Thickness of the lines used to draw a text
                # lineType:     Line type
            cv2.putText(frame, top_1_prediction, (x, y+(h+50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

        # Display the resulting frame
        frame = cv2.resize(frame, (800, 500))
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()