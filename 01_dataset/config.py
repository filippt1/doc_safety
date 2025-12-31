import os

# config file for dataset generation

# output settings
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_DIR = ""

# subdirectories for outputs
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean")
DEGRADED_DIR = os.path.join(OUTPUT_DIR, "degraded")
OCR_DIR = os.path.join(OUTPUT_DIR, "ocr_texts")

# input directories
INPUT_TEXT_DIR = ""  # source text files
TEXTURE_DIR = "textures"  # paper texture images
STAINS_DIR = "stains"  # stain images (white background)

# font settings
FONT_DIR = "/System/Library/Fonts/Supplemental"  # path to font files
FONT_NAMES = [
    "Arial.ttf",
    # "Helvetica Neue.ttf", # no cz support
    # "Calibri.ttf", # no cz support
    "Times New Roman.ttf",
    "Courier New.ttf",
    "Verdana.ttf",
    "Georgia.ttf"
]
FONT_SIZES = [24, 20, 18, 16, 14]
