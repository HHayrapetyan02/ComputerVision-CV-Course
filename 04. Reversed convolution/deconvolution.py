import numpy as np
from scipy.fft import fft2, ifft2, ifftshift, fftshift

def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    center = (size - 1) / 2
    x, y = np.mgrid[:size, :size]
    dist_sq = ((x - center)**2 + (y - center)**2)

    gaussian_kernel = np.exp(-dist_sq / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    gaussian_kernel /= np.sum(gaussian_kernel)
    return gaussian_kernel


def pad_kernel(kernel, shape):
    high, width = shape
    kernel_high, kernel_width = kernel.shape[:2]
    ph, pw = high - kernel_high, width - kernel_width

    padding = [((ph + 1) // 2, ph // 2), ((pw + 1) // 2, pw // 2)]
    return np.pad(kernel, padding)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    h_padded = pad_kernel(h, shape)
    return fft2(ifftshift(h_padded))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros_like(H, dtype=np.complex128)
    mask = (np.abs(H) > threshold)
    H_inv[mask] = 1.0 / H[mask]

    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    H_inv = inverse_kernel(H, threshold)

    F = G * H_inv
    f_transform = np.abs(fftshift(ifft2(F)))
    return f_transform
                          

def wiener_filtering(blurred_img, h, K=1e-4):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    H_conj = np.conj(H)
    H_abs = H_conj * H
    F = (H_conj / (H_abs + K)) * G

    f_inv_transform = np.abs(fftshift(ifft2(F)))
    return f_inv_transform


def mse(img1, img2):
    high, width = img1.shape
    return np.sum((img1 - img2) ** 2) / (high * width)

def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    MAX = 255.0
    MSE = mse(img1, img2)
    if MSE == 0:
        raise ValueError
    return 20 * np.log10(MAX / np.sqrt(MSE))
    
