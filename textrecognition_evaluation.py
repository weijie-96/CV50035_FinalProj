
from PIL import Image
import pytesseract
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

img_folder = ""

for img_name in os.listdir(img_folder):

	# check if it is a folder or a filer
	img_path = os.path.join(img_folder, img_name)
	if not os.path.isfile(img_path):
		continue
	print(img_name)

	# read the image
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# read the text
	text = pytesseract.image_to_string(Image.fromarray(gray))

	# save the output as a .txt file
	with open(os.path.join(img_folder, "%s_text.txt"%(img_name)), "w+") as f:
		f.write(text)
