import os
import cv2
import numpy as np
import tensorflow as tf
from networks import DnCNN, DRUNet, Pix2Pix

CURRENT_MODEL_NAME = ''  # drunet or dncnn or pix2pix
WEIGHTS_PATH = ''  # path to the model weights file

INPUT_DIR = ''  # input (degraded) images directory
OUTPUT_DIR = f'results/inference_{CURRENT_MODEL_NAME}/'  # output (enhanced) directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Setting up {CURRENT_MODEL_NAME.upper()}")

INPUT_SHAPE = (None, 256, 256, 3)  # input shape (batch, H, W, channels); channels=3 for RGB, 1 for grayscale
dummy_input = tf.zeros((1, 256, 256, 3))  # dummy input for building the model

if CURRENT_MODEL_NAME == 'pix2pix':  # for pix2pix, we only need the generator
    temp_gan = Pix2Pix()
    model = temp_gan.generator
    _ = model(dummy_input)  # build the model

    try:
        model.load_weights(WEIGHTS_PATH)
        print(f"Generator weights for {CURRENT_MODEL_NAME} loaded from {WEIGHTS_PATH}")
    except Exception as e:
        print(f"Error loading generator weights: {e}")
        exit()

elif CURRENT_MODEL_NAME in ['dncnn', 'drunet']:  # here, we load the full denoising models

    if CURRENT_MODEL_NAME == 'dncnn':
        model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=17)
    else:
        model = DRUNet()

    _ = model(dummy_input)  # build the model

    try:
        model.load_weights(WEIGHTS_PATH)
        print(f"Model weights for {CURRENT_MODEL_NAME} loaded from {WEIGHTS_PATH}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit()

else:
    raise ValueError(f"Unknown model: {CURRENT_MODEL_NAME}")

print(f"Processing images to {OUTPUT_DIR}...")

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # loading image
    path = os.path.join(INPUT_DIR, filename)
    org_img = cv2.imread(path)
    if org_img is None: continue
    img = cv2.resize(org_img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    # normalization
    if CURRENT_MODEL_NAME == 'pix2pix':
        img = (img / 127.5) - 1.0
    else:
        img = img / 255.0

    # inference
    input_tensor = np.expand_dims(img, axis=0)
    pred = model.predict(input_tensor, verbose=0)[0]

    # denormalization
    if CURRENT_MODEL_NAME == 'pix2pix':
        pred = (pred + 1.0) * 127.5
    else:
        pred = pred * 255.0

    # saving output
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    final_img = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_img)
    print(f"Processed: {filename}")

print("Inference Complete.")
