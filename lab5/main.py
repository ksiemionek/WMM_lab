import cv2
import numpy as np
from matplotlib import pyplot as plt
from functions import calc_entropy, img_histogram, img_differential, dwt


img_original = cv2.imread('./images/girl_col.png', cv2.IMREAD_UNCHANGED)


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
plt.show()

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
plt.show()

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
plt.show()
