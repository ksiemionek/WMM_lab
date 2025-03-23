import cv2
from matplotlib import pyplot as plt
from functions import calcPSNR


img_original = cv2.imread('./images/girl_col.png', cv2.IMREAD_UNCHANGED)

img_yuv = cv2.cvtColor(img_original, cv2.COLOR_BGR2YUV)

img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imwrite('./images/zad2_girl_col_equalized.png', img_equalized)

psnr_value = calcPSNR(img_original, img_equalized)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Obraz oryginalny")
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img_equalized, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Obraz po wyrównaniu histogramu")
axes[0, 1].axis('off')

hist_original = cv2.calcHist([img_original], [0], None, [256], [0, 256])
hist_original = hist_original.flatten()
hist_equalized = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
hist_equalized = hist_equalized.flatten()

axes[1, 0].plot(hist_original)
axes[1, 0].set_title("Histogram - obraz oryginalny")
axes[1, 0].set_xlim([0, 256])
axes[1, 0].grid(True)

axes[1, 1].plot(hist_equalized)
axes[1, 1].set_title("Histogram - po wyrównaniu")
axes[1, 1].set_xlim([0, 256])
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("./images/zad2_histogramy.png")
plt.close()
