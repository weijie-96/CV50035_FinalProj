from PIL import Image
import pytesseract
import argparse
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

img_file = "D:/SUTD_matters/term7/Computer_Vision/final_proj/deblur-gan-adapt/experiment/textoutput/textmotion"

for i in range(1,3):

	img = img_file+ "/%d.jpg"%(i)
	print(img)

	image = cv2.imread(img)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, gray)

	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)

	with open( img_file +"/%d.txt"%(i), "w+") as f:
		f.write(text)
	print(text)



# show the output images
# cv2.imshow("Image", image)
# cv2.imshow("Output", gray)
# cv2.waitKey(0)
