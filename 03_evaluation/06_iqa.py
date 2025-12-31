import os
import cv2
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse


GT_DIR = ''  # path to ground truth images

# paths to degraded/enhanced images
DIRS_TO_TEST = {
    'Degraded (Baseline)': '',
    'Pix2Pix': '',
    'DnCNN': '',
    'DRUNet': ''
}

OUTPUT_CSV = 'results/final_image_quality_summary.csv'


# compute the metrics between ground truth and predicted images
def calculate_metrics(gt_path, pred_path):
    if not os.path.exists(pred_path):
        return None

    img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    img_pred = cv2.imread(pred_path, cv2.IMREAD_COLOR)

    if img_gt is None or img_pred is None:
        return None

    if img_gt.shape != img_pred.shape:
        img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))

    # MSE
    val_mse = mse(img_gt, img_pred)

    # PSNR
    if (val_mse == 0):
        print(f"MSE is zero between {gt_path} and {pred_path}")
        val_psnr = 100  # otherwise PSNR is undefined/infinite
    else:
        val_psnr = psnr(img_gt, img_pred, data_range=255)

    # SSIM
    val_ssim = ssim(img_gt, img_pred, data_range=255, channel_axis=2)

    return val_mse, val_psnr, val_ssim


summary_data = []

gt_files = [f for f in os.listdir(GT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not gt_files:
    print(f"Error: No images found in {GT_DIR}")
    exit()

print(f"Found {len(gt_files)} Ground Truth images. Starting evaluation...\n")

for model_name, model_dir in DIRS_TO_TEST.items():
    print(f"Evaluating: {model_name}...")

    model_mse, model_psnr, model_ssim = [], [], []

    if not os.path.exists(model_dir):
        print(f" Warning: Directory not found ({model_dir}). Skipping.")
        continue

    for filename in gt_files:
        path_gt = os.path.join(GT_DIR, filename)
        path_pred = os.path.join(model_dir, filename)

        results = calculate_metrics(path_gt, path_pred)

        if results:
            m, p, s = results
            model_mse.append(m)
            model_psnr.append(p)
            model_ssim.append(s)

    if model_mse:
        avg_mse = np.mean(model_mse)
        avg_psnr = np.mean(model_psnr)
        avg_ssim = np.mean(model_ssim)

        print(f"  MSE: {avg_mse:.2f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
        summary_data.append([model_name, avg_mse, avg_psnr, avg_ssim])
    else:
        print("  No matching images found.")

# save results to CSV
df = pd.DataFrame(summary_data, columns=['Model', 'MSE', 'PSNR', 'SSIM'])
df = df.sort_values(by='SSIM', ascending=False)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to '{OUTPUT_CSV}'.")
print(df.to_string(index=False))