import cv2
import numpy as np

def calculate_psnr(img1, img2):
    """
    Calculate the PSNR between two images.
    :param img1: First image (numpy array)
    :param img2: Second image (numpy array)
    :return: PSNR value
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Means no difference
    max_pixel = 255.0  # Assuming 8-bit images
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Load images
image1_path = 'I.png'  # Replace with the path to the first image
image3_path = 'T_m2.png'  # Replace with the path to the third image
image4_path = 'T_m3.png'  # Replace with the path to the fourth image
image2_path = 'T.png'  # Replace with the path to the second image


img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)
img3 = cv2.imread(image3_path)
img4 = cv2.imread(image4_path)

# # Ensure the images have the same dimensions
# if img1.shape != img2.shape:
#     print("Error: Images must have the same dimensions!")
# else:
#     psnr_value = calculate_psnr(img1, img2)
#     print(f"PSNR between the two images: {psnr_value:.2f} dB")
psnr_value = calculate_psnr(img1, img3)    
print(f"m1 PSNR between the two images: {psnr_value:.2f} dB")

psnr_value = calculate_psnr(img1, img4)    
print(f"m2 PSNR between the two images: {psnr_value:.2f} dB")

psnr_value = calculate_psnr(img1, img2)
print(f"m3 PSNR between the two images: {psnr_value:.2f} dB")

