import numpy as np
import cv2

def np2Img(np_image, Normalize=True):
    np_image = np.moveaxis(np_image, 0, -1)
    if Normalize:
        normalized = (np_image - np_image.min()) / (
            np_image.max() - np_image.min()) * 255.0
    else:
        normalized = np_image
    normalized = normalized[:, :, [2, 1, 0]]
    normalized = normalized.astype(np.uint8)
    return normalized


def np2Depth(input_tensor, invaild_mask):
    normalized = (input_tensor - 0.02) / (2.0 - 0.02) * 255.0
    normalized = normalized.astype(np.uint8)
    normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)
    normalized[invaild_mask] = 0
    return normalized