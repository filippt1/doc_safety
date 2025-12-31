import os
import cv2
import random
import config
from renderer import TextRenderer
from degrader import Degrader


# before running, fill config.py file
def generate_dataset():
    renderer = TextRenderer()  # initialize renderer
    degrader = Degrader()  # initialize degrader

    # create dirs
    os.makedirs(config.CLEAN_DIR, exist_ok=True)
    os.makedirs(config.DEGRADED_DIR, exist_ok=True)
    os.makedirs(config.OCR_DIR, exist_ok=True)

    # check input dirs
    os.makedirs(config.INPUT_TEXT_DIR, exist_ok=True)
    os.makedirs(config.STAINS_DIR, exist_ok=True)

    input_files = [f for f in os.listdir(config.INPUT_TEXT_DIR) if f.endswith('.txt')]

    if not input_files:
        print(f"No .txt files found in '{config.INPUT_TEXT_DIR}'")
        return

    print(f"Processing {len(input_files)} text files...")

    # process each text file
    count = 0
    for filename in input_files:
        path = os.path.join(config.INPUT_TEXT_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()
        if not raw_text: continue

        # render the text
        align_mode = "left" if random.random() > 0.2 else "center"
        base_text_img, fitting_text = renderer.create_image(raw_text, alignment=align_mode)

        if not fitting_text: continue

        # apply texture
        clean_img = degrader.apply_paper_texture(base_text_img)

        # create degraded version
        degraded_img = clean_img.copy()

        # create text bleed layer
        bleed_layer = None
        if random.random() > 0.6:
            bleed_layer = degrader.create_bleed_layer(base_text_img)

        # blend bleed layer if exists
        if bleed_layer is not None:
            degraded_img = cv2.multiply(degraded_img, bleed_layer, scale=1.0 / 255.0)

        # apply other degradations
        if random.random() > 0.3:
            degraded_img = degrader.apply_stain(degraded_img)

        if random.random() > 0.6:
            degraded_img = degrader.apply_morphology(degraded_img)

        if random.random() > 0.3:
            degraded_img = degrader.apply_blur(degraded_img)

        if random.random() > 0.7:
            degraded_img = degrader.apply_salt_pepper(degraded_img)

        # apply perspective distortion
        # we comment out, as we need pixel-level alignment for training
        # if random.random() > 0.4:
        #     degraded_img = degrader.apply_perspective(degraded_img)

        # save clean and degraded images
        base_name = os.path.splitext(filename)[0] + "_1.jpg"
        cv2.imwrite(os.path.join(config.CLEAN_DIR, base_name), clean_img)
        cv2.imwrite(os.path.join(config.DEGRADED_DIR, base_name), degraded_img)

        # save the OCR ground truth text
        ocr_text_path = os.path.join(config.OCR_DIR, os.path.splitext(base_name)[0] + ".txt")
        with open(ocr_text_path, 'w', encoding='utf-8') as f:
            f.write(fitting_text)

        count += 1
        print(f"Generated pair: {base_name}")

    print(f"Created {count} pairs in '{config.OUTPUT_DIR}'")


generate_dataset()
