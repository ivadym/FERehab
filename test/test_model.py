import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from keras.models import load_model

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Model used
train_model =  "ResNet" # (Inception-v3, Inception-ResNet-v2): Inception,  (ResNet-50): ResNet

# Size of the images
if train_model == "Inception":
	img_width, img_height =	139, 139
elif train_model == "ResNet":
	img_width, img_height =	197, 197

# Test data
test_data_dir = "./../FER-2013/fer2013_test.csv"

# Reinstantiate the fine-tuned model (Also compiling the model using the saved training configuration (unless the model was never compiled))
model = load_model("./../trained_models/ResNet-50.h5")

batch_size = 1

def preprocess_input(x):
	if train_model == "Inception":
		x /= 127.5
		x -= 1.
		return x
	elif train_model == "ResNet":
		x -= 128.8006	# np.mean(train_dataset)
		x /= 64.6497	# np.std(train_dataset)
	return x

def get_data(dataset):

	file_stream = file_io.FileIO(test_data_dir, mode="r")
	data = pd.read_csv(file_stream)
	pixels = data["pixels"].tolist()
	images = np.empty((len(data), img_height, img_width, 3))
	i = 0
	
	for pixel_sequence in pixels:
	    single_image = [float(pixel) for pixel in pixel_sequence.split(" ")]	# Extraction of each image
	    single_image = np.asarray(single_image).reshape(48, 48)					# Dimension: 48x48
	    single_image = resize(single_image, (img_height, img_width), order = 3, mode = "constant") # Bicubic
	    ret = np.empty((img_height, img_width, 3))  
	    ret[:, :, 0] = single_image
	    ret[:, :, 1] = single_image
	    ret[:, :, 2] = single_image
	    images[i, :, :, :] = ret
	    i += 1
	
	images = preprocess_input(images)
	labels = data["emotion"].tolist()

	return images, labels					

images, labels = get_data(test_data_dir)

# Generates output predictions for the input samples
	# x: 			The input data, as a Numpy array
	# batch_size: 	Integer. If unspecified, it will default to 32
# Returns a numpy array of predictions
predictions = model.predict(
	images,
	batch_size	= batch_size)

predicted_classes	= np.argmax(predictions, axis = 1)		# Returns the class (position of the row) of maximum prediction
true_classes 		=  labels 								# Returns the correct classes associated with the predictions
class_names 		= list(["Anger", "Disgust", "Fear", "Happinness", "Sadness", "Surprise", "Neutral"])	# Returns the names of the classes

# Accuracy classification score. In multilabel classification, this function computes subset accuracy: 
# the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true
	# true_classes:			Ground truth (correct) labels
	# predicted_classes:	Predicted labels, as returned by a classifier
	# normalize:			If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly
	#		 				classified samples (default = True)
# If normalize == True, returns the correctly classified samples (float), else it returns the number of correctly classified samples (int).
accuracy = accuracy_score(
	true_classes, 
	predicted_classes, 
	normalize = True)

# Build a text report showing the main classification metrics
	# true_classes:			Ground truth (correct) target values
	# predicted_classes:	Estimated targets as returned by a classifier
	# target_names:			Optional display names matching the labels (same order)
# Returns text summary of the precision, recall and F1 score for each class
report = classification_report(
	true_classes, 
	predicted_classes,
	target_names = class_names)

# Print the result of the evaluation
print("Accuracy:")
print(accuracy)
print("\n")
print("Report:")
print(report)
print("\n")