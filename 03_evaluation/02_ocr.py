import os
import cv2
import pytesseract
import easyocr
from PIL import Image

# from surya.foundation import FoundationPredictor
# from surya.recognition import RecognitionPredictor
# from surya.detection import DetectionPredictor

GT_ROOT = '../dataset/data/dataset_split/test/'  # path to test subset
INFERENCE_ROOT = 'results/inference'  # path to inference results
OUTPUT_ROOT = 'results/ocr'  # path to save OCR results
ENGINES = ['tesseract', 'easyocr']  # OCR engines; we can also use Surya engine, however, it is slow
REFERENCE_FOLDERS = ['clean', 'degraded']

print("Loading OCR Engine...")
reader_easy = easyocr.Reader(['cs', 'en'], gpu=True)


# print("Loading Surya model...")
# foundation_predictor = FoundationPredictor()
# det_predictor = DetectionPredictor()
# rec_predictor = RecognitionPredictor(foundation_predictor)


def run_tesseract(img_path):
    return pytesseract.image_to_string(img_path, lang='ces+eng').strip().replace('\n', ' ')


def run_easyocr(img_path):
    res = reader_easy.readtext(img_path, detail=0)
    return " ".join(res)


# def run_surya(img_path):
#     try:
#         image = Image.open(img_path).convert("RGB")
#         predictions = rec_predictor(image, [det_predictor(image, foundation_predictor)[0]])
#
#         lines = []
#         for text_line in predictions[0].text_lines:
#             lines.append(text_line.text)
#         return " ".join(lines)
#     except Exception as e:
#         print(f"Surya Error on {img_path}: {e}")
#         return ""


# gather all image folders to process
all_folders = []

# reference folders (gt and degraded)
for ref in REFERENCE_FOLDERS:
    full_path = os.path.join(GT_ROOT, ref)
    if os.path.exists(full_path):
        all_folders.append((ref, full_path))

# inference folders (enhanced degraded images)
if os.path.exists(INFERENCE_ROOT):
    for d in os.listdir(INFERENCE_ROOT):
        if d.startswith('inference_'):
            name = d.replace('inference_', '')
            path = os.path.join(INFERENCE_ROOT, d)
            all_folders.append((name, path))

# run every OCR engine on every folder
for model_name, input_path in all_folders:
    print(f"Running OCR on {model_name} output")

    for engine in ENGINES:
        output_folder = os.path.join(OUTPUT_ROOT, f"ocr_{model_name}_{engine}")
        os.makedirs(output_folder, exist_ok=True)
        print(f"Engine: {engine}")

        files = [f for f in sorted(os.listdir(input_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in files:
            img_path = os.path.join(input_path, filename)
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(output_folder, f"{base_name}.txt")

            if os.path.exists(save_path): continue

            text = ""
            if engine == 'tesseract':
                text = run_tesseract(img_path)
            elif engine == 'easyocr':
                text = run_easyocr(img_path)
            # elif engine == 'surya':
            #     text = run_surya(img_path)

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(text)

print("OCR Extraction Complete.")
