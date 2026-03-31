import cv2
import time
import math
import mediapipe as mp
 
# ---------------- CONFIG ----------------
HOLD_DELAY = 0.35
CLICK_COOLDOWN = 0.6
BLINK_FRAMES = 12
 
# ---------------- LOAD ASSETS ----------------
images = [
    cv2.imread("assets/images/img1.jpg"),
    cv2.imread("assets/images/img2.jpg"),
    cv2.imread("assets/images/img3.jpg"),
    cv2.imread("assets/images/img4.jpg"),
    cv2.imread("assets/images/img5.jpg"),
]
 
instruction_power = cv2.imread("assets/images/instruction_pinch.png")
instruction_slider = cv2.imread("assets/images/instruction_slide.png")
 
# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
 
# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
 
# ---------------- STATE ----------------
img_index = 0
power_on = False
last_click = 0
 
blink_left = 0
blink_right = 0
 
# ---------------- WINDOW FULLSCREEN (PGU) ----------------
cv2.namedWindow("Gesture Demo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Gesture Demo",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)
 
# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
 
    cursor_x, cursor_y = None, None
    pinch = False
 
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
 
        idx = hand.landmark[8]
        thumb = hand.landmark[4]
 
        cursor_x = int(idx.x * w)
        cursor_y = int(idx.y * h)
 
        cv2.circle(frame, (cursor_x, cursor_y), 12, (0, 255, 0), -1)
 
        pinch = math.hypot(idx.x - thumb.x, idx.y - thumb.y) < 0.07
 
    # ====================================================
    # POWER PAGE
    # ====================================================
    if not power_on:
        center = (w // 2, h // 2)
        radius = 130
 
        cv2.circle(frame, center, radius, (0, 0, 255), -1)
 
        hover = False
        if cursor_x and abs(cursor_x - center[0]) < radius and abs(cursor_y - center[1]) < radius:
            hover = True
            cv2.circle(frame, center, radius + 10, (0, 255, 255), 5)
 
        cv2.putText(frame, "OFF", (center[0]-45, center[1]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 4)
 
        # Instruction overlay
        if instruction_power is not None:
            ih, iw, _ = instruction_power.shape
            frame[h-ih-20:h-20, w//2-iw//2:w//2+iw//2] = instruction_power
 
        if hover and pinch and time.time() - last_click > CLICK_COOLDOWN:
            power_on = True
            last_click = time.time()
 
        cv2.imshow("Gesture Demo", frame)
        if cv2.waitKey(1) == 27:
            break
        continue
 
    # ====================================================
    # SLIDER PAGE
    # ====================================================
    frame[:] = images[img_index].copy()
 
    # Arrow positions
    left_rect = (20, h//2-60, 100, 120)
    right_rect = (w-120, h//2-60, 100, 120)
 
    # Draw arrows
    left_color = (255,255,255) if blink_left > 0 else (180,180,180)
    right_color = (255,255,255) if blink_right > 0 else (180,180,180)
 
    cv2.putText(frame, "<", (40, h//2+40),
                cv2.FONT_HERSHEY_SIMPLEX, 3, left_color, 6)
    cv2.putText(frame, ">", (w-80, h//2+40),
                cv2.FONT_HERSHEY_SIMPLEX, 3, right_color, 6)
 
    # Instruction overlay (static image)
    if instruction_slider is not None:
        ih, iw, _ = instruction_slider.shape
        frame[h-ih-20:h-20, w//2-iw//2:w//2+iw//2] = instruction_slider
 
    # Click detection
    if cursor_x and pinch and time.time() - last_click > CLICK_COOLDOWN:
 
        if 20 < cursor_x < 120:
            img_index = (img_index - 1) % len(images)
            blink_left = BLINK_FRAMES
            last_click = time.time()
 
        elif w-120 < cursor_x < w-20:
            img_index = (img_index + 1) % len(images)
            blink_right = BLINK_FRAMES
            last_click = time.time()
 
    # Blink countdown
    blink_left = max(0, blink_left - 1)
    blink_right = max(0, blink_right - 1)
 
    cv2.imshow("Gesture Demo", frame)
    if cv2.waitKey(1) == 27:
        break
 
# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
 