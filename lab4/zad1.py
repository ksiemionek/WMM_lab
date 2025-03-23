import cv2
from matplotlib import pyplot as plt
from functions import calcPSNR

masks = [(3, 3), (5, 5), (7, 7)]

img_original = cv2.imread('./images/girl_col.png', cv2.IMREAD_UNCHANGED)
img_noise = cv2.imread('./images/girl_col_noise.png', cv2.IMREAD_UNCHANGED)
img_inoise = cv2.imread('./images/girl_col_inoise.png', cv2.IMREAD_UNCHANGED)

psnr_noise = calcPSNR(img_original, img_noise)
psnr_inoise = calcPSNR(img_original, img_inoise)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
axes[0].set_title("Oryginał")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Szum Gaussowski (PSNR: {psnr_noise:.3f})")
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(img_inoise, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Szum Impulsowy (PSNR: {psnr_inoise:.3f})")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("./images/zad1_oryginaly.png")
plt.close()

for mask in masks:
    img_gaussian_blur = cv2.GaussianBlur(img_noise, mask, 0)
    psnr_gauss = calcPSNR(img_original, img_gaussian_blur)

    img_median_blur = cv2.medianBlur(img_inoise, mask[0])
    psnr_median = calcPSNR(img_original, img_median_blur)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(cv2.cvtColor(img_gaussian_blur, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Filtr Gaussa {mask} (PSNR: {psnr_gauss:.2f})")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_median_blur, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Filtr medianowy {mask} (PSNR: {psnr_median:.2f})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"./images/zad1_noise_maska_{mask[0]}x{mask[1]}.png")
    plt.close()

    img_gaussian_blur_inoise = cv2.GaussianBlur(img_inoise, mask, 0)
    psnr_gauss_inoise = calcPSNR(img_original, img_gaussian_blur_inoise)

    img_median_blur_inoise = cv2.medianBlur(img_inoise, mask[0])
    psnr_median_inoise = calcPSNR(img_original, img_median_blur_inoise)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(cv2.cvtColor(img_gaussian_blur_inoise, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Filtr Gaussa {mask} (PSNR: {psnr_gauss_inoise:.2f})")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_median_blur_inoise, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Filtr medianowy {mask} (PSNR: {psnr_median_inoise:.2f})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"./images/zad1_inoise_maska_{mask[0]}x{mask[1]}.png")
    plt.close()