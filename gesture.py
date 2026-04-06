import cv2
import time
import mediapipe as mp
import os
import numpy as np
from pathlib import Path
import threading
from trixel_3_api.TcmController import TcmController
# =========================
# CONFIG
# =========================
VIDEO_DIR = os.path.expanduser("~/Documents/PHUD/videos")
VIDEO_FILES_DISPLAY = [
   "output1.mp4",
   "output2.mp4",
   "output3.mp4",
   "output4.mp4",
   "output5.mp4",
]
VIDEO_FILES_PROJECTOR = [
   "output1.h264",
   "output2.h264",
   "output3.h264",
   "output4.h264",
   "output5.h264",
]
SCREEN_W = 1920
SCREEN_H = 1080
PINCH_THRESHOLD = 0.035
CLICK_COOLDOWN = 0.5
SMOOTHING = 0.35
# =========================
# PROJECTOR INIT
# =========================
tcm = TcmController(logging_en=False)
tcm.change_ip_address("11.11.11.1")
tcm.projector_write(True)
# =========================
# PROJECTOR CONTROLLER (🔥 FIX)
# =========================
current_video = None
video_lock = threading.Lock()
def projector_controller():
   global current_video
   last_video = None
   while True:
       with video_lock:
           video = current_video
       if video is None:
           time.sleep(0.1)
           continue
       if video != last_video:
           print("Now playing:", video)
           last_video = video
       # 🔥 loops automatically
       tcm.video_stream(video)
# start controller thread
threading.Thread(target=projector_controller, daemon=True).start()
# =========================
# VIDEO CAPTURE (DISPLAY)
# =========================
caps = [
   cv2.VideoCapture(str(Path(VIDEO_DIR)/v))
   for v in VIDEO_FILES_DISPLAY
]
# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap_cam = cv2.VideoCapture(0)
# =========================
# STATE
# =========================
prev_x, prev_y = 0, 0
dragging = False
selected_video = None
last_click = 0
prev_pinch = False
# =========================
# HELPERS
# =========================
def smooth(x, y):
   global prev_x, prev_y
   x = int(prev_x + SMOOTHING * (x - prev_x))
   y = int(prev_y + SMOOTHING * (y - prev_y))
   prev_x, prev_y = x, y
   return x, y
def detect_pinch(idx, thumb):
   dx = idx.x - thumb.x
   dy = idx.y - thumb.y
   dist = (dx*dx + dy*dy) ** 0.5
   return dist < PINCH_THRESHOLD
# =========================
# WINDOW
# =========================
cv2.namedWindow("PHUD", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("PHUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# =========================
# MAIN LOOP
# =========================
while True:
   ret, cam = cap_cam.read()
   if not ret:
       continue
   cam = cv2.flip(cam, 1)
   canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
   # -------------------------
   # HAND TRACKING
   # -------------------------
   rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
   result = hands.process(rgb)
   cursor_x, cursor_y, pinch = None, None, False
   if result.multi_hand_landmarks:
       hand = result.multi_hand_landmarks[0]
       idx = hand.landmark[8]
       thumb = hand.landmark[4]
       cx = int(idx.x * SCREEN_W)
       cy = int(idx.y * SCREEN_H)
       cx, cy = smooth(cx, cy)
       cursor_x, cursor_y = cx, cy
       pinch = detect_pinch(idx, thumb)
   # -------------------------
   # SQUARE GRID
   # -------------------------
   gap = 20
   max_h = int(SCREEN_H * 0.65)
   max_w = (SCREEN_W - gap * 4) // 5
   square_size = min(max_h, max_w)
   total_width = square_size * 5 + gap * 4
   start_x = (SCREEN_W - total_width) // 2
   y = int(SCREEN_H * 0.1)
   video_boxes = []
   # -------------------------
   # DRAW VIDEOS
   # -------------------------
   for i in range(len(caps)):
       cap = caps[i]
       ret, frame = cap.read()
       if not ret:
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
           ret, frame = cap.read()
       if frame is None:
           frame = np.zeros((square_size, square_size, 3), dtype=np.uint8)
       else:
           frame = cv2.resize(frame, (square_size, square_size))
       x = start_x + i * (square_size + gap)
       if cursor_x and x < cursor_x < x+square_size and y < cursor_y < y+square_size:
           cv2.rectangle(frame, (0,0), (square_size, square_size), (0,255,255), 4)
       canvas[y:y+square_size, x:x+square_size] = frame
       video_boxes.append((x, y, square_size, square_size))
   # -------------------------
   # DROP ZONE
   # -------------------------
   drop_y = y + square_size + 40
   drop_h = 120
   drop_hover = False
   if cursor_y and drop_y < cursor_y < drop_y + drop_h:
       drop_hover = True
   color = (0,255,0) if drop_hover else (0,255,255)
   cv2.rectangle(canvas,
                 (start_x, drop_y),
                 (start_x + total_width, drop_y + drop_h),
                 color, 4)
   cv2.putText(canvas, "DROP TO PROJECT",
               (start_x + total_width//2 - 200, drop_y + 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
   # -------------------------
   # CLICK
   # -------------------------
   click = pinch and not prev_pinch
   prev_pinch = pinch
   if click and time.time() - last_click > CLICK_COOLDOWN:
       for i, (x, yb, s, _) in enumerate(video_boxes):
           if cursor_x and x < cursor_x < x+s and yb < cursor_y < yb+s:
               dragging = True
               selected_video = i
       last_click = time.time()
   # -------------------------
   # DRAG VISUAL
   # -------------------------
   if dragging and selected_video is not None and cursor_x:
       cv2.rectangle(canvas,
                     (cursor_x-70, cursor_y-70),
                     (cursor_x+70, cursor_y+70),
                     (0,255,0), 3)
   # -------------------------
   # DROP (🔥 FIXED)
   # -------------------------
   if not pinch and dragging:
       if drop_hover:
           video_path = Path(VIDEO_DIR) / VIDEO_FILES_PROJECTOR[selected_video]
           print("Switching to:", video_path)
           with video_lock:
               current_video = video_path
       dragging = False
       selected_video = None
   # cursor
   if cursor_x:
       cv2.circle(canvas, (cursor_x, cursor_y), 10, (0,255,0), -1)
   cv2.imshow("PHUD", canvas)
   if cv2.waitKey(1) == 27:
       break
cap_cam.release()
cv2.destroyAllWindows()