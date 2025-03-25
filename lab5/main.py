import cv2
import numpy as np
from matplotlib import pyplot as plt
from functions import (calc_entropy, img_histogram,
                       img_differential, dwt, calc_bitrate)

# MONOCHROME

img_mono = cv2.imread('./images/girl_mono.png', cv2.IMREAD_UNCHANGED)
img_hdiff = img_differential(img_mono)

hist_mono = img_histogram(img_mono)
hist_hdiff = img_histogram((img_hdiff+255).astype(np.uint16), 511)

entropy_mono = calc_entropy(hist_mono)
entropy_hdiff = calc_entropy(hist_hdiff)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(img_mono, cmap='gray')
axes[0, 0].set_title(f"Entropia: {entropy_mono:.3f}")
axes[0, 0].axis('off')

axes[0, 1].imshow(img_hdiff, cmap='gray')
axes[0, 1].set_title(f"Entropia: {entropy_hdiff:.3f}")
axes[0, 1].axis('off')

axes[1, 0].plot(hist_mono, color="blue")
axes[1, 0].set_title("Histogram obrazu oryginalnego")
axes[1, 0].set_xlim([0, 255])

axes[1, 1].plot(np.arange(-255, 256, 1), hist_hdiff, color="red")
axes[1, 1].set_title("Histogram obrazu róznicowego")
axes[1, 1].set_xlim([-255, 255])

plt.tight_layout()
plt.savefig('./images/mono_rozn_histogramy.png')

ll, lh, hl, hh = dwt(img_mono)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(ll, cmap='gray')
axes[0, 0].set_title("Pasmo LL")
axes[0, 0].axis('off')

axes[0, 1].imshow(lh, cmap='gray')
axes[0, 1].set_title("Pasmo LH")
axes[0, 1].axis('off')

axes[1, 0].imshow(hl, cmap='gray')
axes[1, 0].set_title("Pasmo HL")
axes[1, 0].axis('off')

axes[1, 1].imshow(hh, cmap='gray')
axes[1, 1].set_title("Pasmo HH")
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('./images/mono_pasma.png')

hist_ll = img_histogram(ll)
hist_lh = img_histogram((lh+255).astype(np.uint16), 511)
hist_hl = img_histogram((hl+255).astype(np.uint16), 511)
hist_hh = img_histogram((hh+255).astype(np.uint16), 511)

entropy_ll = calc_entropy(hist_ll)
entropy_lh = calc_entropy(hist_lh)
entropy_hl = calc_entropy(hist_hl)
entropy_hh = calc_entropy(hist_hh)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(hist_ll)
axes[0, 0].set_title(f"Entropia LL: {entropy_ll:.3f}")
axes[0, 0].set_xlim([0, 255])

axes[0, 1].plot(np.arange(-255, 256, 1), hist_lh)
axes[0, 1].set_title(f"Entropia LH: {entropy_lh:.3f}")
axes[0, 1].set_xlim([-255, 255])

axes[1, 0].plot(np.arange(-255, 256, 1), hist_hl)
axes[1, 0].set_title(f"Entropia HL: {entropy_hl:.3f}")
axes[1, 0].set_xlim([-255, 255])

axes[1, 1].plot(np.arange(-255, 256, 1), hist_hh)
axes[1, 1].set_title(f"Entropia HH: {entropy_hh:.3f}")
axes[1, 1].set_xlim([-255, 255])

plt.tight_layout()
plt.savefig('./images/mono_pasma_hist.png')

bitrate_mono = calc_bitrate('./images/girl_mono.png', img_mono)

compare = [
    entropy_mono,
    entropy_hdiff,
    entropy_ll,
    entropy_lh,
    entropy_hl,
    entropy_hh,
    bitrate_mono
]

labels = [
    "Oryginał",
    "Różnicowy",
    "LL",
    "LH",
    "HL",
    "HH",
    "Przepływność PNG"
]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, compare)
plt.bar_label(bars, fmt=lambda x: f'{x:.3f}')
plt.tight_layout()
plt.savefig('./images/mono_porownanie.png')

# RGB

img_original = cv2.imread('./images/girl_col.png', cv2.IMREAD_UNCHANGED)

image_R = img_original[:, :, 2]
image_G = img_original[:, :, 1]
image_B = img_original[:, :, 0]

hist_original = img_histogram(img_original)
hist_R = img_histogram(image_R)
hist_G = img_histogram(image_G)
hist_B = img_histogram(image_B)

entropy_original = calc_entropy(hist_original)
entropy_R = calc_entropy(hist_R)
entropy_G = calc_entropy(hist_G)
entropy_B = calc_entropy(hist_B)

r_channel = np.zeros_like(img_original)
r_channel[:, :, 2] = image_R

g_channel = np.zeros_like(img_original)
g_channel[:, :, 1] = image_G

b_channel = np.zeros_like(img_original)
b_channel[:, :, 0] = image_B

r_channel = cv2.cvtColor(r_channel, cv2.COLOR_BGR2RGB)
g_channel = cv2.cvtColor(g_channel, cv2.COLOR_BGR2RGB)
b_channel = cv2.cvtColor(b_channel, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(r_channel)
axes[0].set_title(f"Entropia: {entropy_R:.3f}")
axes[0].axis('off')

axes[1].imshow(g_channel)
axes[1].set_title(f"Entropia: {entropy_G:.3f}")
axes[1].axis('off')

axes[2].imshow(b_channel)
axes[2].set_title(f"Entropia: {entropy_B:.3f}")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('./images/rgb_warstwy.png')

plt.figure(figsize=(8, 6))
plt.plot(hist_R, color="red")
plt.plot(hist_G, color="green")
plt.plot(hist_B, color="blue")
plt.xlim([0, 255])
plt.tight_layout()
plt.savefig('./images/rgb_hist.png')

img_yuv = cv2.cvtColor(img_original, cv2.COLOR_BGR2YUV)

image_Y = img_yuv[:, :, 0]
image_U = img_yuv[:, :, 1]
image_V = img_yuv[:, :, 2]

hist_Y = img_histogram(image_Y)
hist_U = img_histogram(image_U)
hist_V = img_histogram(image_V)

entropy_Y = calc_entropy(hist_Y)
entropy_U = calc_entropy(hist_U)
entropy_V = calc_entropy(hist_V)

display_U = np.full(img_yuv.shape, 127).astype(np.uint8)
display_U[:, :, 1] = img_yuv[:, :, 1]

display_V = np.full(img_yuv.shape, 127).astype(np.uint8)
display_V[:, :, 2] = img_yuv[:, :, 2]

# cv2.imwrite('./images/yuv_Y.png', image_Y)
# cv2.imwrite('./images/yuv_U.png', display_U)
# cv2.imwrite('./images/yuv_V.png', display_V)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

axes[0, 0].imshow(image_Y, cmap='gray')
axes[0, 0].set_title(f"Y - Entropia: {entropy_Y:.3f}")
axes[0, 0].axis('off')

display_U_rgb = cv2.cvtColor(display_U, cv2.COLOR_YUV2RGB)
axes[0, 1].imshow(display_U_rgb)
axes[0, 1].set_title(f"U - Entropia: {entropy_U:.3f}")
axes[0, 1].axis('off')

display_V_rgb = cv2.cvtColor(display_V, cv2.COLOR_YUV2RGB)
axes[0, 2].imshow(display_V_rgb)
axes[0, 2].set_title(f"V - Entropia: {entropy_V:.3f}")
axes[0, 2].axis('off')

axes[1, 0].imshow(r_channel)
axes[1, 0].set_title(f"R - Entropia: {entropy_R:.3f}")
axes[1, 0].axis('off')

axes[1, 1].imshow(g_channel)
axes[1, 1].set_title(f"G - Entropia: {entropy_G:.3f}")
axes[1, 1].axis('off')

axes[1, 2].imshow(b_channel)
axes[1, 2].set_title(f"B - Entropia: {entropy_B:.3f}")
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('./images/yuv_rgb_warstwy.png')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(hist_Y, color="black", label="Y")
axes[0].plot(hist_U, color="purple", label="U")
axes[0].plot(hist_V, color="pink", label="V")
axes[0].set_xlim([0, 255])
axes[0].legend()

axes[1].plot(hist_R, color="red")
axes[1].plot(hist_G, color="green")
axes[1].plot(hist_B, color="blue")
axes[1].set_xlim([0, 255])

plt.tight_layout()
plt.savefig('./images/yuv_rgb_hist.png')
