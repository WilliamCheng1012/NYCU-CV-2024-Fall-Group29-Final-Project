import cv2
import lpips
import numpy as np
from scipy.signal import convolve2d
import os
from tqdm import tqdm  # 引入 tqdm 模組

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
        raise ValueError("Input images must have the same dimensions")
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

# 計算所有指標
def calculate_all(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # 轉換為灰度圖像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 計算PSNR
    psnr = calculate_psnr(img1_gray, img2_gray)

    # 計算SSIM
    ssim = compute_ssim(img1_gray, img2_gray)

    # 計算LPIPS
    lpips_value = calculate_lpips(img1, img2)
    
    return psnr, ssim, lpips_value


# 引入進度條計算
def evaluate_dataset(name, C_path, E_path, eval_list, file_extension=".png"):
    print(f'Estimating {name}')
    PSNR = []
    SSIM = []
    LPIPS = []

    for i in tqdm(eval_list, desc=f"Processing {name}"):  # 使用 tqdm 進行進度條展示
        C = os.path.join(C_path, i)
        # E = os.path.join(E_path, i.replace(file_extension, '/t_output.png'))
        E = os.path.join(E_path, i.replace(file_extension, '/errnet.png'))
        psnr, ssim, lpips_value = calculate_all(C, E)
        PSNR.append(psnr)
        SSIM.append(ssim)
        LPIPS.append(lpips_value)

    print(f'{name} - PSNR: {np.mean(PSNR):.4f}, SSIM: {np.mean(SSIM):.4f}, LPIPS: {np.mean(LPIPS):.4f}')
    print('')

# Main Evaluation
print("=====================================")
print("Start evaluating...")

# Synthetic_table2
synthetic_table2_C_path = '../data/testing_set/testdata_reflection_synthetic_table2/transmission_layer'
# synthetic_table2_E_path = 'test_results/synthetic_table2'
synthetic_table2_E_path = '/home/hpc/Project/312510232/Final_Project/DL_TermProject/model_3_code/results/CEILNet_table2/'
synthetic_table2_eval_list = os.listdir(synthetic_table2_C_path)
evaluate_dataset("Synthetic_table2", synthetic_table2_C_path, synthetic_table2_E_path, synthetic_table2_eval_list)

# # SolidObjectDataset
sod_C_path = '../data/testing_set/SIR2/SolidObjectDataset/transmission_layer'
sod_E_path = 'results/solidobject/'
sod_eval_list = os.listdir(sod_C_path)
evaluate_dataset("SolidObjectDataset", sod_C_path, sod_E_path, sod_eval_list)

# # Postcard
postcard_C_path = '../data/testing_set/SIR2/PostcardDataset/transmission_layer'
postcard_E_path = 'results/postcard/'
postcard_eval_list = os.listdir(postcard_C_path)
evaluate_dataset("Postcard", postcard_C_path, postcard_E_path, postcard_eval_list)

# # Wild
wild_C_path = '../data/testing_set/SIR2/Wildscene/transmission_layer'
wild_E_path = 'results/wild/'
wild_eval_list = os.listdir(wild_C_path)
evaluate_dataset("Wild", wild_C_path, wild_E_path, wild_eval_list)

# # Nature
# nature_C_path = '../data/testing_set/Nature/transmission_layer'
# nature_E_path = 'test_results/Nature'
# nature_eval_list = os.listdir(nature_C_path)
# evaluate_dataset("Nature", nature_C_path, nature_E_path, nature_eval_list, file_extension=".jpg")

print("Finish evaluating...")
print("=====================================")
