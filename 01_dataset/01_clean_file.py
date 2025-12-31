import argparse
import os


def clean_file(input_path, overwrite=False):
    forbidden_words = ["chapter", "illustration", "copyright"]  # words to filter out lines

    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' was not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    removed_count = 0
    blank_count = 0

    # line by line processing
    for line in lines:
        # remove blank lines
        if not line.strip():
            blank_count += 1
            continue

        line_lower = line.lower()

        # check for forbidden words
        if any(word in line_lower for word in forbidden_words):
            removed_count += 1
            continue  # skip if forbidden word found

        # remove undesired characters
        clean_line = line.replace('_', '').replace('*)', '').replace('»', '').replace('«', '')
        cleaned_lines.append(clean_line)

    # save
    if overwrite:
        output_path = input_path
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cleaned{ext}"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        print(f"Processed {input_path}'")
        print(f"Removed {blank_count} blank lines")
        print(f"Removed {removed_count} lines containing {forbidden_words}")
        print(f"Saved to: '{output_path}'")

    except Exception as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean text files: removes blank lines, forbidden words, and undesired characters.")
    parser.add_argument("file", help="Path to the .txt file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the original file")

    args = parser.parse_args()
    clean_file(args.file, args.overwrite)