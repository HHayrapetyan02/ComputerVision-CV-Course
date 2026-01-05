import numpy as np
from scipy.signal import convolve, convolve2d

def get_bayer_masks(n_rows, n_cols):
    base_mask = np.array([
        [[False, True, False], [True, False, False]],
        [[False, False, True], [False, True, False]]
    ], dtype=bool)

    repeat_base_mask = np.tile(base_mask, ((n_rows + 1) // 2, (n_cols + 1) // 2, 1))
    bayer_mask = repeat_base_mask[:n_rows, :n_cols, :]
    return bayer_mask


def get_colored_img(raw_img):
    mask = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    return raw_img[:, :, np.newaxis] * mask.astype(np.uint8)


def get_raw_img(colored_img):
    mask = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    return np.sum(colored_img * mask, axis=2).astype(np.uint8)


def bilinear_interpolation(raw_img):
    high, width = raw_img.shape
    mask = get_bayer_masks(high, width)
    
    kernel = np.ones((3, 3))
    result = np.empty((high, width, 3), dtype=np.uint8)
    
    for i in range(3):
        known = raw_img * mask[:, :, i]
        
        padded_known = np.pad(known, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        padded_mask = np.pad(mask[:, :, i].astype(float), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        
        sum_conv = np.zeros_like(known, dtype=float)
        count_conv = np.zeros_like(known, dtype=float)
        
        for j in range(3):
            for k in range(3):
                sum_conv += padded_known[j:j+high, k:k+width] * kernel[j, k]
                count_conv += padded_mask[j:j+high, k:k+width] * kernel[j, k]
      
        result[:,:,i] = np.clip(np.where(mask[:,:,i], known, sum_conv/np.maximum(count_conv,1)), 0, 255).astype(np.uint8)
        
    return result


def create_interpolation_filters():
    green_filter = np.array([
        [ 0.0,  0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0,  2.0,  0.0,  0.0],
        [-1.0,  2.0,  4.0,  2.0, -1.0],
        [ 0.0,  0.0,  2.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  0.0,  0.0]
    ]) / 8.0

    red_blue_green_filter = np.array([
        [ 0.0,  0.0,  0.5,  0.0,  0.0],
        [ 0.0, -1.0,  0.0, -1.0,  0.0],
        [-1.0,  4.0,  5.0,  4.0, -1.0],
        [ 0.0, -1.0,  0.0, -1.0,  0.0],
        [ 0.0,  0.0,  0.5,  0.0,  0.0]
    ]) / 8.0

    red_blue_cross_filter = np.array([
        [ 0.0,  0.0, -1.5,  0.0,  0.0],
        [ 0.0,  2.0,  0.0,  2.0,  0.0],
        [-1.5,  0.0,  6.0,  0.0, -1.5],
        [ 0.0,  2.0,  0.0,  2.0,  0.0],
        [ 0.0,  0.0, -1.5,  0.0,  0.0]
    ]) / 8.0

    return green_filter, red_blue_green_filter, red_blue_cross_filter


def create_masks(mask):
    color_rows = np.any(mask, axis=1)[:, np.newaxis]
    color_cols = np.any(mask, axis=0)[np.newaxis, :]

    return color_rows, color_cols


def calculate_convolutions(raw_img):
    _, rb_green_filter, rb_cross_filter = create_interpolation_filters()
    rb_green_conv = convolve2d(raw_img, rb_green_filter, mode="same")
    rb_green_transposed_conv = convolve2d(raw_img, rb_green_filter.T, mode="same")
    rb_cross_conv = convolve2d(raw_img, rb_cross_filter, mode="same")
    
    return rb_green_conv, rb_green_transposed_conv, rb_cross_conv


def color_interpolation(raw_img, red_channel, blue_channel, red_mask, blue_mask):
    red_rows, red_cols = create_masks(red_mask)
    blue_rows, blue_cols = create_masks(blue_mask)
    rb_green_conv, rb_green_transposed_conv, rb_cross_conv = calculate_convolutions(raw_img)

    red_in_green_blue = np.logical_and(red_rows, blue_cols)
    red_in_blue_green = np.logical_and(blue_rows, red_cols)
    red_in_blue_blue = np.logical_and(blue_rows, blue_cols)
    red_channel[red_in_green_blue] = rb_green_conv[red_in_green_blue]
    red_channel[red_in_blue_green] = rb_green_transposed_conv[red_in_blue_green]
    red_channel[red_in_blue_blue] = rb_cross_conv[red_in_blue_blue]
    
    blue_in_blue_red = np.logical_and(blue_rows, red_cols)
    blue_in_red_blue = np.logical_and(red_rows, blue_cols)
    blue_in_red_red = np.logical_and(red_rows, red_cols)
    blue_channel[blue_in_blue_red] = rb_green_conv[blue_in_blue_red]
    blue_channel[blue_in_red_blue] = rb_green_transposed_conv[blue_in_red_blue]
    blue_channel[blue_in_red_red] = rb_cross_conv[blue_in_red_red]

    return red_channel, blue_channel


def improved_interpolation(raw_img):
    height, width = raw_img.shape
    masks = get_bayer_masks(height, width)
    colored_img = get_colored_img(raw_img.astype(np.float64))
    
    red_channel, green_channel, blue_channel = [colored_img[:, :, i] for i in range(3)]
    red_mask, green_mask, blue_mask = [masks[:, :, i] for i in range(3)]
    
    green_filter, _, _ = create_interpolation_filters()
    green_interpolated = convolve2d(raw_img, green_filter, mode="same")

    green_channel[red_mask | blue_mask] = green_interpolated[red_mask | blue_mask]
    red_channel, blue_channel = color_interpolation(raw_img, red_channel, blue_channel, red_mask, blue_mask)
    
    result = np.stack([red_channel, green_channel, blue_channel], axis=2)
    return np.clip(result, 0, 255).astype(np.uint8)


def mse_score(img_pred, img_gt):
    high, width, dim = img_pred.shape
    return np.sum((img_pred - img_gt) ** 2) / (high * width * dim)


def convert_float(img):
    return img.astype(np.float64)


def compute_psnr(img_pred, img_gt):
    conv_pred = convert_float(img_pred)
    conv_gt = convert_float(img_gt)

    mse = mse_score(conv_pred, conv_gt)
    if mse == 0:
        raise ValueError
    PSNR = 10 * np.log10(np.max(conv_gt) ** 2 / mse)
    return PSNR


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")
