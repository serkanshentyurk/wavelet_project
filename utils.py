import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image

def load_image(path):
    '''
    Load images from the given path Makes sure the image contains even-numbered pixels (for practical reasons)
    Returns the image as numpy array
    '''
    sample_image_rgb = Image.open(path)
    sample_image = np.array(sample_image_rgb.convert('L'))
    if sample_image.shape[0]%2 == 1:
        sample_image = sample_image[:-1]
    if sample_image.shape[1]%2 == 1:
        sample_image = sample_image[:,:-1]
    return sample_image

def piecewise_smooth_function(x):
    '''
    Returns y y=f(x) where f(x) is the given function at the project pdf
    '''
    y = (2 + np.cos(x))*np.abs(x)*np.sign(x-1)
    return y

def add_noise(data, epsilon = 2):
    '''
    Add noise * epsilon to the data
    Retursn noise-added-data
    '''
    noise = epsilon * np.random.randn(*data.shape)
    return noise + data

def modify_denoised_coefs(coefs):
    ''''''
    converted_list = [coefs[0]]
    for i in range(1, len(coefs)):
        converted_list.append(tuple(coefs[i]))
    return converted_list

def denoising_with_wavedec(data, 
                           wavelet = 'db2', 
                           level = 5, 
                           threshold = 5, 
                           image = True, 
                           mode = 'soft'):
    '''
    Returns coef, coef_thresholded_hard, coef_thresholded_soft, f_soft_denoised, f_hard_denoised
    '''
    coeffs = None
    coeffs_thresholded_hard = None
    coeffs_thresholded_soft = None
    f_soft_denoised = None
    f_hard_denoised = None
    
    if image:
        coeffs = pywt.wavedec2(data, wavelet, level = level)
        if mode == 'soft':
            coeffs_thresholded_soft = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            converted_list_soft = modify_denoised_coefs(coeffs_thresholded_soft)
            f_soft_denoised = pywt.waverec2(converted_list_soft, wavelet)
        elif mode == 'hard':
            coeffs_thresholded_hard = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
            converted_list_hard = modify_denoised_coefs(coeffs_thresholded_hard)
            f_hard_denoised = pywt.waverec2(converted_list_hard, wavelet)
                        
        elif mode == 'both':
            coeffs_thresholded_soft = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            converted_list_soft = modify_denoised_coefs(coeffs_thresholded_soft)                
            f_soft_denoised = pywt.waverec2(converted_list_soft, wavelet)
            coeffs_thresholded_hard = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
            converted_list_hard = modify_denoised_coefs(coeffs_thresholded_hard)
            f_hard_denoised = pywt.waverec2(converted_list_hard, wavelet)
        else:
            print('Wrong mode selection')
            return
    else:
        coeffs = pywt.wavedec(data, wavelet, level = level)
        if mode == 'soft':
            coeffs_thresholded_soft = [pywt.threshold(c, threshold, mode = 'soft') for c in coeffs]
            f_soft_denoised = pywt.waverec(coeffs_thresholded_soft, wavelet)
            
        elif mode == 'hard':
            coeffs_thresholded_hard = [pywt.threshold(c, threshold, mode = 'hard') for c in coeffs]
            f_hard_denoised = pywt.waverec(coeffs_thresholded_hard, wavelet)
            
        elif mode == 'both':
            coeffs_thresholded_soft = [pywt.threshold(c, threshold, mode = 'soft') for c in coeffs]
            f_soft_denoised = pywt.waverec(coeffs_thresholded_soft, wavelet)
            coeffs_thresholded_hard = [pywt.threshold(c, threshold, mode = 'hard') for c in coeffs]
            f_hard_denoised = pywt.waverec(coeffs_thresholded_hard, wavelet)   
                     
    return coeffs, coeffs_thresholded_hard, coeffs_thresholded_soft, f_soft_denoised, f_hard_denoised

def threshold_swt(original, copy, threshold, mode):
    for key in original[0]:
        copy[0][key] = pywt.threshold(original[0][key], value = threshold, mode=mode)
    return copy

def denoising_with_swt(data, 
                       wavelet = 'db2', 
                       threshold = 20, 
                       level = 1, 
                       start_level = 0, 
                       mode = 'soft', 
                       image = True):
    '''
    Returns denoised_image_array_soft, denoised_image_array_hard
    '''
    
    denoised_image_array_soft = None
    denoised_image_array_hard = None
    
    if image:
        coeffs = pywt.swtn(data, wavelet = wavelet, level = level, start_level = start_level)
        converted = coeffs.copy()
        
        if mode == 'soft':
            converted_list_soft = threshold_swt(coeffs, converted, threshold, mode = 'soft')
            denoised_image_array_soft = pywt.iswtn(converted_list_soft, wavelet)
        elif mode == 'hard':
            converted_list_hard = threshold_swt(coeffs, converted, threshold, mode = 'hard')
            denoised_image_array_hard = pywt.iswtn(converted_list_hard, wavelet)
        elif mode == 'both':
            converted_list_soft = threshold_swt(coeffs, converted, threshold, mode = 'soft')
            denoised_image_array_soft = pywt.iswtn(converted_list_soft, wavelet)
            converted_list_hard = threshold_swt(coeffs, converted, threshold, mode = 'hard')
            denoised_image_array_hard = pywt.iswtn(converted_list_hard, wavelet)
        else:
            print('Wrong mode!')
    else:
        coeffs = pywt.swt(data, wavelet = wavelet, level = level)
        if mode == 'soft':
            coeffs_thresholded_soft = pywt.threshold(coeffs, threshold, mode='soft')
            coeffs_thresholded_soft = [tuple(coeffs_thresholded_soft[0])]
            denoised_image_array_soft = pywt.iswt(coeffs_thresholded_soft, wavelet = wavelet)

        elif mode == 'hard':
            coeffs_thresholded_hard = pywt.threshold(coeffs, threshold, mode='hard')
            coeffs_thresholded_hard = [tuple(coeffs_thresholded_hard[0])]
            denoised_image_array_hard = pywt.iswt(coeffs_thresholded_hard, wavelet = wavelet)

        elif mode == 'both':
            coeffs_thresholded_soft = pywt.threshold(coeffs, threshold, mode='soft')
            coeffs_thresholded_soft = [tuple(coeffs_thresholded_soft[0])]
            coeffs_thresholded_hard = pywt.threshold(coeffs, threshold, mode='hard')
            coeffs_thresholded_hard = [tuple(coeffs_thresholded_hard[0])]
            denoised_image_array_hard = pywt.iswt(coeffs_thresholded_hard, wavelet = wavelet)
            denoised_image_array_soft = pywt.iswt(coeffs_thresholded_soft, wavelet = wavelet)
        else:
            print('Wrong mode!')

    return denoised_image_array_soft, denoised_image_array_hard

def plot_signals(x, original, noisy, soft_denoised, hard_denoised, 
                 threshold = 'NA', 
                 wavelet = 'NA', 
                 epsilon = 'NA', 
                 mse_soft = 'NA', 
                 mse_hard = 'NA'):
    '''
    Plot original signal, noisy signal and denoised signal
    '''
    if mse_soft != 'NA':
        mse_soft = round(mse_soft,4)
    if mse_hard != 'NA':
        mse_hard = round(mse_hard,4)
        
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(x, original)
    plt.title('Original Signal')

    plt.subplot(2, 2, 2)
    plt.plot(x, noisy)
    plt.title('Noisy Signal')

    plt.subplot(2, 2, 3)
    plt.plot(x, soft_denoised, label = 'Denoised')
    plt.plot(x, original, 'r', label = 'Truth')
    plt.legend()
    plt.title(f'Soft Thresholding Denoised Signal\nMSE: {mse_soft}')

    plt.subplot(2, 2, 4)
    plt.plot(x, hard_denoised, label = 'Denoised')
    plt.plot(x, original, 'r', label = 'Truth')
    plt.legend()
    plt.title(f'Hard Thresholding Denoised Signal\nMSE: {mse_hard}')

    plt.suptitle(f'Denoisig with Noise {epsilon}: Wavelet = {wavelet} - Threshold = {threshold} - Borders = Symmetric')
    plt.tight_layout()
    plt.show()
    
def plot_images(original, noisy = None, denoised_soft = None, denoised_hard = None):
    '''
    Plot original image, noisy image, and denoised images
    '''
    if type(noisy) == type(None):
        noisy = np.zeros(original.shape)
    if type(denoised_soft) == type(None):
        denoised_soft = np.zeros(original.shape)
    if type(denoised_hard) == type(None):
        denoised_hard = np.zeros(original.shape)
        
    # Display or save the original, noisy, and denoised images
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap = 'gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(denoised_soft, cmap='gray')
    plt.title('Denoised Image - Mode: Soft')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(denoised_hard, cmap='gray')
    plt.title('Denoised Image - Mode: Hard')
    plt.axis('off')
    plt.show()
    
def calculate_snr(original_image, denoised_image):
    '''
    Inputs original data and denoised data
    Returns signal-to-noise ratio
    '''
    original_array = np.array(original_image).astype(float)
    denoised_array = np.array(denoised_image).astype(float)

    signal_power = np.sum(original_array ** 2)
    noise_power = np.sum((original_array - denoised_array) ** 2)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def create_mask(image, random = True, mask_size = (512,512), true_count = 50544):
    '''
    Create a mask for the condition given at the project pdf or randomly allocated mask
    '''
    if random:
        mask = np.zeros(mask_size, dtype=bool)
        indices = np.random.choice(np.prod(mask_size), true_count, replace=False)
        mask.flat[indices] = True
    else:
        mask = np.zeros_like(image)
        mask[::10, :] = 1
        mask[:, ::10] = 1
        mask = mask > 0
    return mask


def wavelet_inpainting(A, mask, B0, max_iter, wavelet = 'db2', level = 5, threshold = 20):
    '''
    Returns denoised image: wavedec_soft, wavedec_hard, swt_soft, swt_hard
    '''
    A[mask] = B0
    A_wavedec_soft = A.copy()
    A_wavedec_hard = A.copy()
    A_swt_soft = A.copy()   
    A_swt_hard = A.copy()


    for _ in range(max_iter):        
        _, _, _, image_wavedec_soft, _ = denoising_with_wavedec(A_wavedec_soft,
                                                                wavelet = wavelet, 
                                                                level = level, 
                                                                threshold = threshold, 
                                                                mode = 'soft', 
                                                                image = True)
        
        _, _, _, _, image_wavedec_hard = denoising_with_wavedec(A_wavedec_hard,
                                                                wavelet = wavelet, 
                                                                level = level, 
                                                                threshold = threshold, 
                                                                mode = 'hard', 
                                                                image = True)
        
        image_swt_soft, _ = denoising_with_swt(A_swt_soft, wavelet = wavelet, 
                                               threshold = threshold, 
                                               level = level, 
                                               start_level = 0,
                                               mode = 'soft', 
                                               image = True)
        
        _, image_swt_hard = denoising_with_swt(A_swt_hard, wavelet = wavelet, 
                                               threshold = threshold, 
                                               level = level,
                                               start_level = 0, 
                                               mode = 'hard', 
                                               image = True)
        
        A_wavedec_soft[mask] = image_wavedec_soft[mask]
        A_wavedec_hard[mask] = image_wavedec_hard[mask]
        A_swt_soft[mask] = image_swt_soft[mask]
        A_swt_hard[mask] = image_swt_hard[mask]

    return A_wavedec_soft, A_wavedec_hard, A_swt_soft, A_swt_hard