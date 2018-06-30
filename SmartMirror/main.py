import os
import csv
import cv2
import time
import datetime
import random
import _thread
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model

class Application(tk.Frame):
	def __init__(self, master = None):
		self.video_capture = cv2.VideoCapture(0)
		super().__init__(master)
		self.pack()
		self.configure_geometry()
		self.LST_logo = "./icons/LST_logo.png"
		# Emotion recognition variables
		self.cascPath = "./../trained_models/haarcascade_frontalface_alt.xml"
		self.haar_cascade = cv2.CascadeClassifier(self.cascPath)
		self.model = load_model("./../trained_models/ResNet-50.h5")
		self.model._make_predict_function()
		self.emotions = ["ENFADO", "ASCO", "MIEDO", "ALEGRIA", "TRISTEZA", "SORPRESA", "NEUTRALIDAD"]
		self.avaible_emotions = [0, 3, 5]
		self.emotion_treshold = 0.2
		self.img_width, self.img_height = 197, 197
		self.instructions = None
		self.recognition = False
		self.hits = 0
		self.failures = 0
		self.percentage = ""
		self.cap_user = None
		self.frame = None
		# Save to csv
		self.filepath = "./logs/FER_results.csv"

		self.check_file()
		self.create_widgets()

	def check_file(self):
		if not os.path.exists(os.path.dirname(self.filepath)):
			os.makedirs(os.path.dirname(self.filepath))

	def write_file(self, data):
		with open(self.filepath, "a") as f:
		    writer = csv.writer(f)
		    writer.writerow(data)	

	def create_widgets(self):
		self.logo = Image.open(self.LST_logo)
		self.logo = self.logo.resize((100, 30), Image.ANTIALIAS)
		self.logo = self.flatten_alpha(self.logo)
		self.tk_logo = ImageTk.PhotoImage(self.logo)
		self.label = tk.Label(image = self.tk_logo, bg = "black")
		self.label.image = self.tk_logo # It's necessary keep a reference
		self.label.place(relx = .5, rely = 0.05, anchor = "center")

		self.button_start = tk.Button(self.master, text = " START ", command = self.start_game, highlightbackground = "black")
		self.button_start.place(relx = .45, rely = .95, anchor = "center")

		self.button_stop = tk.Button(self.master, text = " STOP ", command = self.stop_game, highlightbackground = "black")
		self.button_stop.place(relx = .55, rely = .95, anchor = "center")

		self.txt_instr = tk.Label(self.master, text = "Bienvenido, pulse el botón de START para comenzar",  font = ("Helvetica", "30", "bold"),
		    bg = 'black', fg = 'white', justify = "center")
		self.txt_instr.place(relx = .5, rely = .17, anchor = "center")
		self.txt_instr.after(100, self.update_instructions)

		self.txt_hits = tk.Label(self.master, text = "",  font = ("Helvetica", "30", "bold"),
		    bg = 'black', fg = 'green', justify = "center")
		self.txt_hits.place(relx = .2, rely = .40, anchor = "center")

		self.txt_failures = tk.Label(self.master, text = "",  font = ("Helvetica", "30", "bold"),
		    bg = 'black', fg = 'red', justify = "center")
		self.txt_failures.place(relx = .8, rely = .40, anchor = "center")

		self.txt_percentage = tk.Label(self.master, text = "",  font = ("Helvetica", "30", "bold"),
			bg = 'black', fg = 'green', justify = "center")
		self.txt_percentage.place(relx = .5, rely = .25, anchor = "center")
		self.txt_percentage.after(100, self.update_percentage)

		self.img_user = tk.Label(self.master)
		self.img_user.place(relx = .5, rely = .60, anchor = "center")
		self.img_user.after(100, self.update_img_user)

		_thread.start_new_thread(self.capture_loop, ())
		_thread.start_new_thread(self.reco_loop, ())

	def update_instructions(self):
		self.txt_instr.configure(text = self.instructions)
		self.txt_instr.text = self.txt_instr
		self.txt_instr.after(100, self.update_instructions);

	def update_img_user(self):
		self.img_user.configure(image = self.cap_user)
		self.img_user.img = self.cap_user
		self.img_user.after(250, self.update_img_user)

	def update_percentage(self):
		self.txt_percentage.configure(text = self.percentage)
		self.txt_percentage.text = self.txt_percentage
		self.txt_percentage.after(100, self.update_percentage);

	def capture_loop(self):
		while True:
			ret, self.frame = self.video_capture.read()
			self.frame = cv2.flip(self.frame, 1)
			rgb_frame = self.frame[...,::-1] # BGR to RGB
			rgb_frame = cv2.resize(rgb_frame, (600, 400))
			gimg = Image.fromarray(rgb_frame)
			self.cap_user = ImageTk.PhotoImage(image = gimg)

	def reco_loop(self):
		while True:
			time.sleep(4)
			if self.recognition:
				# target = random.randint(0, 6)
				target = random.choice(self.avaible_emotions)
				self.instructions = "Exprese " + self.emotions[target]
				time.sleep(0.2)
				counter = 0
				millis_ini = int(round(time.time() * 1000))
				while (not self.recognize_emotion(target) and counter <= 50):
					counter += 1
					if not self.recognition:
						self.percentage = ""
						break
				millis_end = int(round(time.time() * 1000))

				if self.recognition:
					if counter < 50:
						self.hits += 1
						self.instructions = "¡Lo ha conseguido!"
						reaction_t = str((millis_end - millis_ini)/1000)
						self.percentage = "Tiempo: %s seg." % reaction_t
						now = datetime.datetime.now()
						data = ["ACIERTO", self.emotions[target], reaction_t, datetime.date.today(), str('%02d:%02d'%(now.hour, now.minute))]
						self.write_file(data)
					else:
						self.failures += 1
						self.percentage = ""
						self.instructions = "¡Inténtelo de nuevo!"
						now = datetime.datetime.now()
						data = ["FALLO", self.emotions[target], "NULL" , datetime.date.today(), str('%02d:%02d'%(now.hour, now.minute))]
						self.write_file(data)

	def preprocess_input(self, image):
		image = cv2.resize(image, (self.img_width, self.img_height))  # Resizing images for the trained model ResNet-50 (197x197)
		ret = np.empty((self.img_height, self.img_width, 3)) 
		ret[:, :, 0] = image
		ret[:, :, 1] = image
		ret[:, :, 2] = image
		x = np.expand_dims(ret, axis = 0)   # (1, 197, 197, 3)
		x -= 128.8006 # np.mean(train_dataset)
		x /= 64.6497 # np.std(train_dataset)
		return x

	def recognize_emotion(self, target_emotion):
		gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 

		# Detects objects of different sizes in the input image 
		#The detected objects are returned as a list of rectangles
		faces = self.haar_cascade.detectMultiScale(
		    gray_frame,
		    scaleFactor     = 1.1,
		    minNeighbors    = 5,
		    minSize         = (30, 30))
		
		for (x, y, w, h) in faces:
		    ROI_gray = gray_frame[y:y+h, x:x+w] # Extraction of the region of interest (face) from the frame
		    emotion = self.preprocess_input(ROI_gray)

		    prediction = self.model.predict(emotion)[0]
		    self.percentage = str(int(prediction[target_emotion]*100)) + "%"

		    if (prediction[target_emotion] >= self.emotion_treshold):
		        return True
		    else:
		        return False

	def configure_geometry(self):
	    self.master.title("LifeSTech")
	    screen_width, screen_height = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
	    self.master.geometry(("%dx%d+0+0") % (screen_width, screen_height))
	    # Background color
	    self.master.configure(bg = "black")

	def flatten_alpha(self, img):
	    alpha = img.split()[-1]  # Pull off the alpha layer
	    ab = alpha.tobytes()  # Original 8-bit alpha
	    
	    checked = []  # Create a new array to store the cleaned up alpha layer bytes
	    
	    # Walk through all pixels and set them either to 0 for transparent or 255 for opaque fancy pants
	    transparent = 50  # Change to suit your tolerance for what is and is not transparent
	    p = 0
	    for pixel in range(0, len(ab)):
	        if ab[pixel] < transparent:
	            checked.append(0)  # Transparent
	        else:
	            checked.append(255)  # Opaque
	        p += 1
	    
	    mask = Image.frombytes("L", img.size, bytes(checked))
	    img.putalpha(mask)
	    
	    return img

	def start_game(self):
		self.recognition = True
		self.instructions = "Imite la EXPRESIÓN FACIAL indicada"
		self.percentage = ""
		self.hits = 0
		self.failures = 0

		self.txt_hits.configure(text = "")
		self.txt_hits.text = self.txt_hits

		self.txt_failures.configure(text = "")
		self.txt_failures.text = self.txt_failures

	def stop_game(self):
		self.recognition = False
		self.instructions = "EJERCICIO FINALIZADO \n Presiona START para volver a comenzar"
		self.percentage = ""

		self.txt_hits.configure(text = "ACIERTOS \n %d" % (self.hits))
		self.txt_hits.text = self.txt_hits

		self.txt_failures.configure(text = "FALLOS \n %d" % (self.failures))
		self.txt_failures.text = self.txt_failures

root = tk.Tk()
app = Application(master = root)
app.mainloop()