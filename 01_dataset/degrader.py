import os
import random
import numpy as np
import cv2
import config


# degrader class to apply various degradations to images
class Degrader:
    def __init__(self):
        # load textures and stains
        self.textures = self._load_images(config.TEXTURE_DIR)
        self.stains = self._load_images(config.STAINS_DIR)

    def _load_images(self, folder):
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found.")
            return []
        return [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ]

    # get a random crop from a list of images
    def _get_random_crop(self, image_list):
        if not image_list: return None

        path = random.choice(image_list)
        img = cv2.imread(path)
        if img is None: return None

        h, w, _ = img.shape

        x = random.randint(0, w - config.IMG_WIDTH)
        y = random.randint(0, h - config.IMG_HEIGHT)

        return img[y:y + config.IMG_HEIGHT, x:x + config.IMG_WIDTH]

    # apply paper texture to an image
    def apply_paper_texture(self, image):
        texture = self._get_random_crop(self.textures)

        if texture is None:
            # fallback noise texture
            texture = np.full(image.shape, (240, 240, 220), dtype=np.uint8)
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            texture = cv2.add(texture, noise)

        # blend the texture with the image
        return cv2.multiply(texture, image, scale=1.0 / 255.0)

    # apply stain to an image
    def apply_stain(self, image):
        stain_patch = self._get_random_crop(self.stains)
        if stain_patch is None: return image

        # pick an opacity for the stain
        opacity = np.random.uniform(0.3, 0.75)

        # create a white patch of the same size
        white_patch = np.full_like(stain_patch, 255)

        # dilute the stain with white based on opacity
        adjusted_stain = cv2.addWeighted(stain_patch, opacity, white_patch, 1.0 - opacity, 0)

        # blend the stain with the image
        return cv2.multiply(image, adjusted_stain, scale=1.0 / 255.0)

    # create text bleed through layer
    @staticmethod
    def create_bleed_layer(clean_text_image):
        # flip and blur
        bleed = cv2.flip(clean_text_image, 1)
        k = random.choice([3, 5])
        # sigma = random.uniform(0, 2)
        bleed = cv2.GaussianBlur(bleed, (k, k), 0)

        # get an intensity of the bleed
        alpha = random.uniform(0.25, 0.7)
        beta = 1.0 - alpha
        white = np.full(bleed.shape, 255, dtype=np.uint8)

        # blend with white
        return cv2.addWeighted(bleed, alpha, white, beta, 0)

    # apply morphological operations
    @staticmethod
    def apply_morphology(image):
        op = random.choice([cv2.MORPH_ERODE, cv2.MORPH_DILATE, cv2.MORPH_OPEN])
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(image, op, kernel)

    # apply salt and pepper noise
    @staticmethod
    def apply_salt_pepper(image):
        output = np.copy(image)
        prob = np.random.uniform(0.005, 0.02)
        h, w = image.shape[:2]
        num_pixels = h * w

        num_pepper = int(prob * num_pixels * 0.5)
        y_coords = np.random.randint(0, h, num_pepper)
        x_coords = np.random.randint(0, w, num_pepper)
        output[y_coords, x_coords] = 0

        num_salt = int(prob * num_pixels * 0.5)
        y_coords = np.random.randint(0, h, num_salt)
        x_coords = np.random.randint(0, w, num_salt)
        output[y_coords, x_coords] = 255

        return output

    # apply Gaussian blur
    @staticmethod
    def apply_blur(image):
        k = random.choice([3, 5])
        sigma = random.uniform(1.0, 3.0)
        return cv2.GaussianBlur(image, (k, k), sigma)

    # apply perspective distortion
    @staticmethod
    def apply_perspective(image):
        rows, cols, _ = image.shape
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        s = random.randint(5, 20)
        pts2 = np.float32([
            [random.randint(0, s), random.randint(0, s)],
            [cols - random.randint(0, s), random.randint(0, s)],
            [random.randint(0, s), rows - random.randint(0, s)],
            [cols - random.randint(0, s), rows - random.randint(0, s)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        avg = np.mean(image, axis=(0, 1))
        return cv2.warpPerspective(image, M, (cols, rows), borderValue=avg)
