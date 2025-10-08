import os
import cv2
import matplotlib.pyplot as plt
import pytesseract

# image_path = "C:/Users/vijey.anbarasan/g500/sample_receipt/g500_sample_receipt_1.png"
image_path = "C:/Users/vijey.anbarasan/g500/sample_receipt/1000087843.jpg"

os.environ['TCL_LIBRARY'] = r'C:\Program Files\Python313\tcl\tcl8.6'

original_image = cv2.imread(image_path)

gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
 

def is_blurry(img, threshold=100):
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


blurry, blur_score = is_blurry(gray)

denoised = cv2.fastNlMeansDenoising(gray, h=30)

thresh = cv2.adaptiveThreshold(
    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\vijey.anbarasan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
extracted_text = pytesseract.image_to_string(thresh)
print("📄 Extracted Text:\n")
print(extracted_text)

images = [original_image, gray, denoised, thresh]
titles = ["Original", "Grayscale", "Denoised", "Thresholded"]


# show images
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    cmap = 'gray' if i > 0 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis('off')

plt.suptitle(f"Blurry: {'Yes' if blurry else 'No'} (Score: {blur_score:.2f})", fontsize=14)
plt.tight_layout()
plt.show()
