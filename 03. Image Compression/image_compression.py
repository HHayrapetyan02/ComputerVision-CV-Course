import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Отцентруем каждую строчку матрицы
    rows_mean = np.mean(matrix, axis=1)
    matrix_mean = matrix - rows_mean[:, None]
    # Найдем матрицу ковариации
    cov_matrix = np.cov(matrix_mean)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    # Посчитаем количество найденных собственных векторов
    count_eig_vec = eig_vec.shape[1]
    # Сортируем собственные значения в порядке убывания
    eig_val_sorted = np.argsort(eig_val)[::-1]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eig_vec_sorted = eig_vec[:, eig_val_sorted]
    # Оставляем только p собственных векторов
    eig_vec_p = eig_vec_sorted[:, :p]
    # Проекция данных на новое пространство
    proj_matrix = np.dot(eig_vec_p.T, matrix_mean)

    return eig_vec_p, proj_matrix, rows_mean


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        
        eig_vec, proj_matrix, rows_mean = comp
        decompressed = np.dot(eig_vec, proj_matrix) + rows_mean[:, None]
        result_img.append(decompressed)

    result_img = np.stack(result_img, axis=-1)
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    return result_img


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed_j = pca_compression(img[:, :, j], p)
            compressed.append(compressed_j)

        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    transform = np.array([
        [0.299, 0.587, 0.114],
        [-0.1687, -0.3313, 0.5],
        [0.5, -0.4187, -0.0813]
    ])

    yCbCr = np.dot(img, transform.T)
    yCbCr[:, :, 1:] += 128

    return np.clip(yCbCr, 0, 255).astype(np.uint8)


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    if img.dtype == np.uint8:
        img = img.astype(np.float64)

    yCbCr = img.copy()
    yCbCr[:, :, 1:] -= 128

    inverse_transform = np.array([
        [1, 0, 1.402],
        [1, -0.34414, -0.71414],
        [1, 1.77, 0]
    ])

    rgb = np.dot(yCbCr, inverse_transform.T)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    yCbCr = rgb2ycbcr(rgb_img)

    Cb_blurred = gaussian_filter(yCbCr[..., 1], sigma=10)
    Cr_blurred = gaussian_filter(yCbCr[..., 2], sigma=10)
    reversed_img = ycbcr2rgb(np.dstack((yCbCr[..., 0], Cb_blurred, Cr_blurred)))
    reversed_full = np.clip(reversed_img, 0, 255).astype(np.uint8)

    plt.imshow(reversed_full)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    yCbCr = rgb2ycbcr(rgb_img)

    y_blurred = gaussian_filter(yCbCr[..., 0], sigma=10)
    reversed_img = ycbcr2rgb(np.dstack((y_blurred, yCbCr[..., 1], yCbCr[..., 2])))
    reversed_full = np.clip(reversed_img, 0, 255).astype(np.uint8)

    plt.imshow(reversed_full)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """

    component_blurred = gaussian_filter(component, sigma=10)
    return component_blurred[::2, ::2]


def alpha(u):
    return 1/np.sqrt(2) if u == 0 else 1

def cosine_sum(x, u):
    return np.cos((2*x + 1) * u * np.pi/16)

def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    n = 8
    dct_block = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            SUM = 0
            for x in range(n):
                for y in range(n):
                    SUM += block[x][y] * cosine_sum(x, u) * cosine_sum(y, v)
            dct_block[u][v] = 1/4 * alpha(u) * alpha(v) * SUM

    return dct_block                        


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    return np.round(block / quantization_matrix)


def scale_factor(q):
    if q < 50:
        return 5000 / q
    elif q < 100:
        return 200 - 2 * q
    else:
        return 1 

def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100
    s = scale_factor(q)

    Quality_factor = np.floor((50 + s * default_quantization_matrix) / 100)
    Quality_factor[Quality_factor == 0] = 1
    
    return Quality_factor


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    n = block.shape[0]
    res = []
    
    for i in range(2 * n - 1):
        if i < n:
            if i % 2 == 0:
                for j in range(i, -1, -1):
                    res.append(block[j, i - j])
            else:
                for j in range(0, i + 1):
                    res.append(block[j, i - j])
        else:
            if i % 2 == 0:
                for j in range(n - 1, i - n, -1):
                    res.append(block[j, i - j])
            else:
                for j in range(i - n + 1, n):
                    res.append(block[j, i - j])
    
    return res


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    res = []
    count_zero = 0

    for val in zigzag_list:
        if val == 0:
            count_zero += 1
        else:
            if count_zero > 0:
                res += [0, count_zero]
                count_zero = 0
            res += [val]
    if count_zero > 0:
        res += [0, count_zero]

    return res            



def split_into_blocks(matrix, block_shape):
    high, width = matrix.shape
    x, y = block_shape

    total_blocks = (high // x) * (width // y)
    blocks = np.zeros((total_blocks, x, y), dtype=matrix.dtype)

    index = 0
    for i in range(0, high, x):
        for j in range(0, width, y):
            blocks[index] = matrix[i:i+x, j:j+y]
            index += 1

    return blocks


def DQZC(block, quantization_matrix):
    return compression(zigzag(quantization(dct(block), quantization_matrix)))


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    # Переходим из RGB в YCbCr
    yCbCr = rgb2ycbcr(img)
    y, Cb, Cr = yCbCr[..., 0], yCbCr[..., 1], yCbCr[..., 2]
    # Уменьшаем цветовые компоненты
    Cb, Cr = downsampling(Cb), downsampling(Cr)
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    y = split_into_blocks(y, (8, 8)) - 128
    Cb = split_into_blocks(Cb, (8, 8)) - 128
    Cr = split_into_blocks(Cr, (8, 8)) - 128
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    compressed_y = [DQZC(block, quantization_matrixes[0]) for block in y]
    compressed_Cb = [DQZC(block, quantization_matrixes[1]) for block in Cb]
    compressed_Cr = [DQZC(block, quantization_matrixes[1]) for block in Cr]

    return [compressed_y, compressed_Cb, compressed_Cr]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    res = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i] == 0:
            res += [0] * int(compressed_list[i+1])
            i += 2
        else:
            res += [compressed_list[i]]
            i += 1
    return res



def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    n = 8
    index = 0
    block = np.zeros((n, n), dtype=type(input[0]) if input else float)
   
    for i in range(2 * n - 1):
        if i < n:
            if i % 2 == 0:
                for j in range(i, -1, -1):
                    if index < len(input):
                        block[j, i - j] = input[index]
                        index += 1
            else:
                for j in range(0, i + 1):
                    if index < len(input):
                        block[j, i - j] = input[index]
                        index += 1
        else:
            if i % 2 == 0:
                for j in range(n - 1, i - n, -1):
                    if index < len(input):
                        block[j, i - j] = input[index]
                        index += 1
            else:
                for j in range(i - n + 1, n):
                    if index < len(input):
                        block[j, i - j] = input[index]
                        index += 1
    
    return block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    n = 8
    inv_dct_block = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            SUM = 0
            for u in range(n):
                for v in range(n):
                    SUM += alpha(u) * alpha(v) * block[u][v] * cosine_sum(x, u) * cosine_sum(y, v)
            inv_dct_block[x][y] = 1/4 * SUM
    
    return np.round(inv_dct_block)            


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [2 * A, 2 * B]
    """

    columns = np.repeat(component, 2, axis=1)
    full_component = np.repeat(columns, 2, axis=0)
    return full_component


def inverse_split_into_blocks(blocks):
    num_blocks, block_h, block_w = blocks.shape
    blocks_per_side = int(np.sqrt(num_blocks))
    blocks_grid = blocks.reshape(blocks_per_side, blocks_per_side, block_h, block_w)
    
    rows = [np.hstack([blocks_grid[i, j] for j in range(blocks_per_side)]) 
            for i in range(blocks_per_side)]
    image = np.vstack(rows)

    return image


def inverse_DQZC(block, quantization_matrix):
    return inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(block)), quantization_matrix))


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    [Y, Cb, Cr] = result
    Y = [inverse_DQZC(y, quantization_matrixes[0]) for y in Y]
    Cb = [inverse_DQZC(cb, quantization_matrixes[1]) for cb in Cb]
    Cr = [inverse_DQZC(cr, quantization_matrixes[1]) for cr in Cr]

    Y = inverse_split_into_blocks(np.array(Y)) + 128
    Cb = inverse_split_into_blocks(np.array(Cb)) + 128
    Cr = inverse_split_into_blocks(np.array(Cr)) + 128

    Cb, Cr = upsampling(Cb), upsampling(Cr)

    RGB = ycbcr2rgb(np.dstack([Y, Cb, Cr]))
    return np.clip(RGB, 0, 255).astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        y_quantization = own_quantization_matrix(y_quantization_matrix, p)
        color_quantization = own_quantization_matrix(color_quantization_matrix, p)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img_restored = jpeg_decompression(compressed, img.shape, matrixes)

        axes[i // 3, i % 3].imshow(img_restored)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
