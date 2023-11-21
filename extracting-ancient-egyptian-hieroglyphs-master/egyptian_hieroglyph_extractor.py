import cv2
import os
import PIL 
from PIL import Image



def process_and_save_images(input_path, output_folder):
    # Load image, grayscale, Gaussian blur, Otsu's threshold, dilate
    image = cv2.imread(input_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours, obtain bounding box coordinates, and extract ROI
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    image_number = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
        ROI = original[y:y + h, x:x + w]
        output_path = os.path.join(output_folder, "ROI_{}.png".format(image_number))
        cv2.imwrite(output_path, ROI)
        image_number += 1
