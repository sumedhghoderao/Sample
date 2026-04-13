import cv2
import numpy as np
import os
import time
import shutil

# ---------------- CONFIG ----------------
SCREEN_W = 800
SCREEN_H = 480
NUM_SCREENS = 3 
FPS = 30
FRAME_DELAY = 1.0 / FPS

TEXT = "UNO MINDA"
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 30
SCROLL_SPEED = 5

BACKGROUND_COLOR = (0, 0, 0)
BOTTOM_MARGIN = 80
SHADOW_OFFSET = 8

OUTPUT_DIR = "output"
# ----------------------------------------

LETTER_COLORS = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
]

# ---------- CLEAN OLD OUTPUT ----------
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)
for i in range(NUM_SCREENS):
    os.makedirs(f"{OUTPUT_DIR}/screen_{i+1}")
# -------------------------------------


def get_font_scale():
    scale = 1
    while True:
        (_, h), _ = cv2.getTextSize(TEXT, FONT, scale, THICKNESS)
        if h >= SCREEN_H - 100:
            return scale - 0.1
        scale += 0.1


def draw_text(canvas, x, y, scale):
    cursor_x = x
    for i, ch in enumerate(TEXT):
        (w, _), _ = cv2.getTextSize(ch, FONT, scale, THICKNESS)
        color = LETTER_COLORS[i % len(LETTER_COLORS)]

        # shadow
        cv2.putText(canvas, ch,
                    (cursor_x + SHADOW_OFFSET, y + SHADOW_OFFSET),
                    FONT, scale, (0, 0, 0),
                    THICKNESS, cv2.LINE_AA)

        # front
        cv2.putText(canvas, ch,
                    (cursor_x, y),
                    FONT, scale, color,
                    THICKNESS, cv2.LINE_AA)

        cursor_x += w + 10


def main():
    font_scale = get_font_scale()

    text_width = sum(
        cv2.getTextSize(ch, FONT, font_scale, THICKNESS)[0][0] + 10
        for ch in TEXT
    )

    TOTAL_W = SCREEN_W * NUM_SCREENS
    x = TOTAL_W
    y = SCREEN_H - BOTTOM_MARGIN

    frame_id = 0

    while True:
        start = time.time()

        canvas = np.full((SCREEN_H, TOTAL_W, 3),
                         BACKGROUND_COLOR, dtype=np.uint8)

        draw_text(canvas, x, y, font_scale)

        for i in range(NUM_SCREENS):
            part = canvas[:, i * SCREEN_W:(i + 1) * SCREEN_W]

            # SHOW
            cv2.imshow(f"Screen {i+1}", part)

            # SAVE only current image
            filename = f"{OUTPUT_DIR}/screen_{i+1}/current.png"
            cv2.imwrite(filename, part)

        x -= SCROLL_SPEED
        if x < -text_width:
            x = TOTAL_W

        frame_id += 1

        elapsed = time.time() - start
        if FRAME_DELAY - elapsed > 0:
            time.sleep(FRAME_DELAY - elapsed)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()