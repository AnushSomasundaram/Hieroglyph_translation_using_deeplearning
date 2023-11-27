import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os , shutil
import PIL 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torchvision.models as models

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

from keytotext import pipeline


data_path = r"/Users/software/Downloads/hieroglyphics_nlp/Code_image_augmentation/augmented_images_dataset"

# Define the transformation to apply to the images (e.g., resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize if needed
])

# Create an instance of ImageFolder and apply the transformation
dataset = (ImageFolder(root=data_path, transform=transform))

# Create a data loader to load the images in batches during training or evaluation
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Function 1: Example function to process the image
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



def predict_vgg16_single(image_path,model_path):
    # load custom vgg model trained on azhars pc
    
    model = models.vgg16(pretrained=False, num_classes=1000)
    model.classifier[6] = nn.Linear(4096, 673)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
    #preprocessing the image
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_image = Image.open(image_path).convert('RGB')
    
    input_data = preprocess(input_image)
    input_data = input_data.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_data)
        
    #Get the predicted class index
    
    _ , predicted_idx = torch.max(outputs,1)
    predicted_index = predicted_idx.item()
    
    return predicted_index




# Function 2: Example function to recognize the hieroglyphs


def process_images_in_folder(input_folder, model_path):
    # Loop through all files in the input folder
    hieroglyph_unicodes = []
    for filename in os.listdir(input_folder):
        # Construct the full path for the input image
        image_path = os.path.join(input_folder, filename)

        # Predict and print the result
        predicted_index = predict_vgg16_single(image_path, model_path)
        hieroglyph_unicodes.append(dataset.classes[predicted_index])
    return hieroglyph_unicodes

def translate_glyphs(unicode_list):
    hieroglyphs_unicodes = " ".join([str(item) for item in unicode_list])
    text = "Translate hieroglyph unicode sequence to English: "+ hieroglyphs_unicodes
    tokenizer = AutoTokenizer.from_pretrained("AnushS/hieroglyph_unicode_translator_t5_small")
    inputs = tokenizer(text, return_tensors="pt").input_ids
    model = AutoModelForSeq2SeqLM.from_pretrained("AnushS/hieroglyph_unicode_translator_t5_small")
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    translated_keywords = str(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return translated_keywords

def delete_files(folder):
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
        



# Streamlit app
def main():
    
    
    st.title("Hieroglyph Translation App")
    nlp = pipeline("k2t")
    
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_location = "./temp_image.png"
        with open(temp_location, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Process the image using Function 1
        process_and_save_images(temp_location,"/Users/software/Downloads/hieroglyphics_nlp/streamlit_dashboard/extracted_images/")

        # Analyze the image using Function 2
        hieroglyphs_recognized = process_images_in_folder("/Users/software/Downloads/hieroglyphics_nlp/streamlit_dashboard/extracted_images/","/Users/software/Downloads/hieroglyphics_nlp/hieroglyph recognition/vgg-16/hieroglyph_vgg_16-mps.pth")

        # translate hieroglyphs
        
        translated_keywords = translate_glyphs(hieroglyphs_recognized)
        
        # Display the processed image
        #st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Display the analysis result
        st.write("Recognized Hieroglyphs: ", hieroglyphs_recognized)

        st.write("Keywords Generated: " , translated_keywords)
        
        #st.write("Sentences Generated: " , nlp(translated_keywords))

        delete_files("/Users/software/Downloads/hieroglyphics_nlp/streamlit_dashboard/extracted_images")
    

if __name__ == "__main__":
    main()
