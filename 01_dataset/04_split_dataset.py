import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("")  # path to the full dataset
# expected structure:
# dataset_full/
#  - clean/
#  - degraded/
#  - ocr_texts/

DEST_DIR = Path("")  # path to save the split dataset

# subfolders to process
FOLDERS = ["clean", "degraded", "ocr_texts"]

# split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# seed for reproducibility
SEED = 42


def split_data():
    random.seed(SEED)

    # check source directory
    base_source = SOURCE_DIR / FOLDERS[0]
    if not base_source.exists():
        print(f"Error: Source folder not found: {base_source}")
        return

    # list files
    print(f"Scanning '{base_source}'...")
    all_files = [f.name for f in base_source.iterdir() if
                 f.is_file() and not f.name.startswith('.')]  # skip hidden files
    total_files = len(all_files)

    if total_files == 0:
        print("No files found in source directory.")
        return

    # shuffle files and split
    random.shuffle(all_files)

    train_end = int(total_files * TRAIN_RATIO)
    val_end = int(total_files * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:]
    }

    print(f"Found {total_files} files.")
    print("Split summary:")
    print(f" Train: {len(splits['train'])}")
    print(f" Val: {len(splits['val'])}")
    print(f" Test: {len(splits['test'])}")

    # process splits
    for split_name, files in splits.items():
        print(f"Processing {split_name.upper()} split...")

        for filename in files:
            for folder in FOLDERS:
                src_folder_path = SOURCE_DIR / folder
                dest_folder_path = DEST_DIR / split_name / folder
                dest_folder_path.mkdir(parents=True, exist_ok=True)

                # identify source file
                src_file = src_folder_path / filename

                if not src_file.exists():
                    if folder == 'ocr_texts':
                        stem = src_file.stem
                        for ext in ['.txt', '.json', '.csv']:
                            potential_file = src_folder_path / (stem + ext)
                            if potential_file.exists():
                                src_file = potential_file
                                break

                    if not src_file.exists():
                        print(f"Warning: Missing corresponding file in '{folder}': {filename}")
                        continue

                # copy file
                shutil.copy2(src_file, dest_folder_path / src_file.name)

    print("Dataset split successfully.")


split_data()
