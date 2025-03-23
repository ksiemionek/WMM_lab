import cv2
import numpy as np
from matplotlib import pyplot as plt

img_original = cv2.imread("./images/girl_col.png")

img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

weights = [-0.5, -1, -1.5, -2, -3]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

axes[0].imshow(img_original_rgb)
axes[0].set_title("Obraz oryginalny")
axes[0].axis('off')

for i, weight in enumerate(weights):
    img_laplace = cv2.Laplacian(img_original, cv2.CV_64F)
    image = np.asarray(img_original, np.float64)
    img_out = cv2.addWeighted(image, 1, img_laplace, weight, 0)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    axes[i+1].imshow(img_out_rgb)
    axes[i+1].set_title(f"Waga = {weight}")
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig("./images/zad3_wyostrzone.png")
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

axes[0].imshow(img_original_rgb)
axes[0].set_title("Obraz oryginalny")
axes[0].axis('off')

for i, weight in enumerate(weights):
    img_gauss = cv2.GaussianBlur(img_original, (3, 3), 0)
    img_laplace = cv2.Laplacian(img_gauss, cv2.CV_64F)
    image = np.asarray(img_original, np.float64)
    img_out = cv2.addWeighted(image, 1, img_laplace, weight, 0)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    axes[i+1].imshow(img_out_rgb)
    axes[i+1].set_title(f"Waga = {weight} (wygładzone)")
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig("./images/zad3_wyostrzone_gauss.png")
plt.show()

best_weight = -3

img_laplace = cv2.Laplacian(img_original, cv2.CV_64F)
img_gauss = cv2.GaussianBlur(img_original, (3, 3), 0)
img_laplace_smooth = cv2.Laplacian(img_gauss, cv2.CV_64F)

image = np.asarray(img_original, np.float64)
img_out_no_smooth = cv2.addWeighted(image, 1, img_laplace, best_weight, 0)
img_out_no_smooth = np.clip(img_out_no_smooth, 0, 255).astype(np.uint8)
img_out_smooth = cv2.addWeighted(image, 1, img_laplace_smooth, best_weight, 0)
img_out_smooth = np.clip(img_out_smooth, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(img_original_rgb)
axes[0].set_title("Obraz oryginalny")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_out_no_smooth, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Waga = {best_weight}")
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(img_out_smooth, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Waga = {best_weight} (wygładzone)")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("./images/zad3_porownanie.png")
plt.show()
