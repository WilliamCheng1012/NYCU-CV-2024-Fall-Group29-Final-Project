
import cv2
import lpips
import numpy as np
from scipy.signal import convolve2d
import os
from tqdm import tqdm

# PSNR
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# SSIM
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.04, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=0.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))

# LPIPS
loss_fn = lpips.LPIPS(net='alex')
def calculate_lpips(image1, image2):
    tensor1 = lpips.im2tensor(image1)
    tensor2 = lpips.im2tensor(image2)
    lpips_value = loss_fn(tensor1, tensor2)
    return lpips_value.item()

# Calculate all metrics
def calculate_all(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR
    psnr = calculate_psnr(img1_gray, img2_gray)

    # Calculate SSIM
    ssim = compute_ssim(img1_gray, img2_gray)

    # Calculate LPIPS
    lpips_value = calculate_lpips(img1, img2)

    return psnr, ssim, lpips_value

main_path = '/home/hpc/Project/312510232/Final_Project/DL_TermProject/data/testing_set'
main_r_path = './results'

# Estimation function with progress bar
def estimate_metrics(category, correct_path, estimate_path, file_extension):
    print(f'Estimating {category}')
    eval_list = os.listdir(correct_path)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for i in tqdm(eval_list, desc=f'Processing {category}'):
        correct = os.path.join(correct_path, i)
        estimate = os.path.join(estimate_path, i.replace(file_extension, '_fake_Ts_03.png'))
        psnr, ssim, lpips_value = calculate_all(correct, estimate)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_value)

    print(f'PSNR: {np.mean(psnr_values):.2f}', f'SSIM: {np.mean(ssim_values):.4f}', f'LPIPS: {np.mean(lpips_values):.4f}')
    print('')

# Categories and paths
datasets = [
    ("test", f"{main_path}/test_m4",
     f"{main_r_path}/testdata_reflection_synthetic_table2/IBCLN/test_final/images", ".png"),
    ("synthetic_table2", f"{main_path}/testdata_reflection_synthetic_table2/transmission_layer", 
     f"{main_r_path}/testdata_reflection_synthetic_table2/IBCLN/test_final/images", ".png"),
    ("SolidObjectDataset", f"{main_path}/SIR2/SolidObjectDataset/transmission_layer", 
     f"{main_r_path}/SolidObjectDataset/IBCLN/test_final/images", ".png"),
    ("PostcardDataset", f"{main_path}/SIR2/PostcardDataset/transmission_layer", 
     f"{main_r_path}/PostcardDataset/IBCLN/test_final/images", ".png"),
    ("Wildscene", f"{main_path}/SIR2/Wildscene/transmission_layer", 
     f"{main_r_path}/Wildscene/IBCLN/test_final/images", ".png"),
    ("Nature_PNG", f"{main_path}/Nature_PNG/transmission_layer", 
     f"{main_r_path}/Nature_PNG/IBCLN/test_final/images", ".png")
]

# Process each dataset
for category, correct_path, estimate_path, file_extension in datasets:
    estimate_metrics(category, correct_path, estimate_path, file_extension)