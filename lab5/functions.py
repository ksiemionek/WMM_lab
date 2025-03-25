import numpy as np
import cv2
import os


def printi(img, img_title="image"):
    print(
        f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, "
        f"wartości: {img.min()} - {img.max()}"
    )


def calc_entropy(hist):
    pdf = hist/hist.sum()
    entropy = -sum([x*np.log2(x) for x in pdf if x != 0])
    return entropy


def img_histogram(image, size=255):
    hist_image = cv2.calcHist([image], [0], None, [size], [0, size])
    hist_image = hist_image.flatten()
    return hist_image


def img_differential(image):
    img_tmp1 = image[:, 1:]
    img_tmp2 = image[:, :-1]
    image_hdiff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0,
                                  dtype=cv2.CV_16S)
    image_hdiff_0 = cv2.addWeighted(image[:, 0], 1, 0, 0, -127,
                                    dtype=cv2.CV_16S)
    image_hdiff = np.hstack((image_hdiff_0, image_hdiff))
    return image_hdiff


def dwt(img):
    maskL = np.array([0.02674875741080976, -0.01686411844287795,
                      -0.07822326652898785, 0.2668641184428723,
                      0.6029490182363579, 0.2668641184428723,
                      -0.07822326652898785, -0.01686411844287795,
                      0.02674875741080976])
    maskH = np.array([0.09127176311424948, -0.05754352622849957,
                      -0.5912717631142470, 1.115087052456994,
                      -0.5912717631142470, -0.05754352622849957,
                      0.09127176311424948])

    bandLL = cv2.sepFilter2D(img,         -1, maskL, maskL)[::2, ::2]
    bandLH = cv2.sepFilter2D(img, cv2.CV_16S, maskL, maskH)[::2, ::2]
    bandHL = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskL)[::2, ::2]
    bandHH = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskH)[::2, ::2]

    return bandLL, bandLH, bandHL, bandHH


def calc_bitrate(out_file_name, image):
    return 8*os.stat(out_file_name).st_size/(image.shape[0]*image.shape[1])
