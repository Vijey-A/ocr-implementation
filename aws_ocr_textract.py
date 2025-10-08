import os

import boto3
import cv2
os.environ['TCL_LIBRARY'] = r'C:\Program Files\Python313\tcl\tcl8.6'

textract = boto3.client('textract',
                        aws_access_key_id='',
                        aws_secret_access_key='',
                        region_name='')


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def save_temp_image(image_array, path='processed_image.png'):
    cv2.imwrite(path, image_array)
    return path


def extract_text_from_aws_textract(image_path):
    pass


def process():
    image_path = "C:/Users/vijey.anbarasan/g500/sample_receipt/1000087843.jpg"
    processed = preprocess_image(image_path)
    processed_path = save_temp_image(processed)
    text = extract_text_from_aws_textract(processed_path)
    print("\n--- Extracted Text ---")
    print(text)


if __name__ == "__main__":
    process()
