import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config


# renderer class to create text images
class TextRenderer:
    def __init__(self):
        # load font names and sizes
        self.font_names = config.FONT_NAMES
        self.font_sizes = config.FONT_SIZES

    # load font with fallback
    def get_font(self, font_name, size):
        try:
            path = os.path.join(config.FONT_DIR, font_name)
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()

    # create image from text
    def create_image(self, text, alignment="left"):
        image = Image.new("RGB", (config.IMG_WIDTH, config.IMG_HEIGHT), color="white")
        draw = ImageDraw.Draw(image)

        # select a random font
        font_name = random.choice(self.font_names)
        font_size = random.choice(self.font_sizes)
        font = self.get_font(font_name, font_size)

        # create margins
        margin = random.randint(10, 20)
        max_w = config.IMG_WIDTH - (2 * margin)
        max_h = config.IMG_HEIGHT - (2 * margin)

        line_height = font_size * 1.2
        input_lines = text.split('\n')

        drawn_lines = []
        current_h = 0

        # fill and truncate lines to fit
        for line in input_lines:
            if (current_h + line_height) > max_h:
                break

            while len(line) > 0:
                bbox = draw.textbbox((0, 0), line, font=font)
                w = bbox[2] - bbox[0]
                if w <= max_w:
                    break
                else:
                    line = line[:-1]  # truncate last character

            # add line if not empty
            if line.strip():
                drawn_lines.append(line)
                current_h += line_height
            elif not line:
                drawn_lines.append("")
                current_h += line_height

        # actual text drawn
        actual_text = "\n".join(drawn_lines).strip()

        # drawing calculations
        total_drawn_h = len(drawn_lines) * line_height

        # apply vertical jitter
        max_start_y = config.IMG_HEIGHT - margin - int(total_drawn_h)
        start_y = random.randint(margin, max(margin, max_start_y))

        current_y = start_y

        # draw each line
        for line in drawn_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]

            if alignment == "center":
                start_x = (config.IMG_WIDTH - line_w) // 2
            else:
                start_x = margin + random.randint(0, 5)

            # apply ink color variation
            val = random.randint(30, 60)
            ink_color = (val, val, val)

            draw.text((start_x, current_y), line, font=font, fill=ink_color)
            current_y += line_height

        # return image and ground truth text
        return np.array(image), actual_text
