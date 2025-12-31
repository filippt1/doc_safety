import os
import textwrap

input_folder = ""  # source folder
output_folder = ""  # destination folder
chars_per_line = 25  # width of the text
lines_per_file = 10  # max height of the text block


# function to process and paginate text files
def process_and_paginate_files(input_dir, output_dir, char_limit=40, max_lines_per_file=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # go through input texts
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):

            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]

            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # flatten the text
                flat_text = " ".join(content.split())

                # wrap the text
                wrapped_lines = textwrap.wrap(flat_text, width=char_limit, break_long_words=False)

                # divide the text into chunks
                chunks = [wrapped_lines[i:i + max_lines_per_file]
                          for i in range(0, len(wrapped_lines), max_lines_per_file)]

                # save the chunks
                if len(chunks) == 0:
                    continue

                elif len(chunks) == 1:
                    output_text = "\n".join(chunks[0])
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output_text)
                    print(f"Processed: {filename} (1 file)")

                else:
                    for index, chunk in enumerate(chunks):
                        output_text = "\n".join(chunk)
                        part_filename = f"{base_name}_part{index + 1}.txt"
                        output_path = os.path.join(output_dir, part_filename)

                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(output_text)
                    print(f"Processed: {filename} (Split into {len(chunks)} files)")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


process_and_paginate_files(input_folder, output_folder, chars_per_line, lines_per_file)
